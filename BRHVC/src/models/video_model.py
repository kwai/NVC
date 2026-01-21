# Copyright (c) 2024 Microsoft Corporation.
# Copyright (c) 2025 Kuaishou Ltd. and/or its affiliates.
#
# This file has been modified by Kuaishou Ltd. and/or its affiliates. on 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/microsoft/DCVC/blob/main/LICENSE.txt.
#
# The modified part from Kuaishou Ltd. and/or its affiliates. is released under the BSD 3-Clause Clear License.

# Copyright 2025 Kuaishou Ltd. and/or its affiliates.
# All rights reserved.
# Licensed under the BSD 3-Clause Clear License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://choosealicense.com/licenses/bsd-3-clause-clear/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
from torch import nn
import numpy as np

from .common_model import CompressionModel
from .video_net import ME_Spynet, ResBlock, UNet, bilinearupsacling, bilineardownsacling, \
    get_hyper_enc_dec_models, flow_warp
from .layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample, DepthConv
from ..utils.stream_helper import get_downsampled_shape, get_padding_size, encode_p, decode_p, filesize, \
    get_state_dict

import torch.nn.functional as F




g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x+3+2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.repeat(1, self.offset_num, 1, 1)
        x = x.view(B * self.group_num * self.offset_num, C // self.group_num, H, W)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, x, context, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, context), dim=1))
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, x, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        mv = self.dec_3(out)
        return mv, feature

## Line 196-425 is modified by Kuaishou
class FB_ContextFusion_mix_Enc(nn.Module):
    def __init__(self, ch1, ch2, ch3, inplace=False):
        super().__init__()
        self.fusion1 = DepthConvBlock(ch1, ch1, inplace=inplace)
        self.fusion2 = DepthConvBlock(ch2, ch2, inplace=inplace)
        self.fusion3 = DepthConvBlock(ch3, ch3, inplace=inplace)

        self.pix_w_maker = DepthConvBlock(9, ch1, stride=1, inplace=inplace)
        self.pix_down1 = DepthConv(ch1, ch1, depth_kernel=3, stride=1, inplace=inplace)
        self.pix_down2 = DepthConv(ch1, ch1, depth_kernel=3, stride=1, inplace=inplace)
        self.pix_down3 = DepthConv(ch1, ch2, depth_kernel=3, stride=2, inplace=inplace)
        self.adp1 = DepthConvBlock(ch1, ch1*3, inplace=inplace)
        self.adp2 = DepthConvBlock(ch1*2, ch2*3, stride=2, inplace=inplace)
        self.adp3 = DepthConvBlock(ch2*2, ch3*3, stride=2, inplace=inplace)
    
    def forward(self, frame, warp_frame_b, warp_frame_f, context1_b, context2_b, context3_b, context1_f, context2_f, context3_f):
        pix_w = self.pix_w_maker(torch.cat([frame, warp_frame_b, warp_frame_f], dim=1))

        pix_w1 = self.pix_down1(pix_w)
        out1 = self.adp1(pix_w1)
        scale1_b, scale1_f, bias1 = torch.chunk(out1, 3, dim=1)
        out1 = (1+scale1_b)*context1_b + (1+scale1_f)*context1_f + bias1
        out1 = self.fusion1(out1)

        pix_w2 = self.pix_down2(pix_w)
        out2 = self.adp2(torch.cat([pix_w2, out1], dim=1))
        scale2_b, scale2_f, bias2 = torch.chunk(out2, 3, dim=1)
        out2 = (1+scale2_b)*context2_b + (1+scale2_f)*context2_f + bias2
        out2 = self.fusion2(out2)

        pix_w3 = self.pix_down3(pix_w)
        out3 = self.adp3(torch.cat([pix_w3, out2], dim=1))
        scale3_b, scale3_f, bias3 = torch.chunk(out3, 3, dim=1)
        out3 = (1+scale3_b)*context3_b + (1+scale3_f)*context3_f + bias3
        out3 = self.fusion3(out3)
        return out1, out2, out3
        
class FB_ContextFusion_mix_Dec(nn.Module):
    def __init__(self, ch1, ch2, ch3, inplace=False, slope=0.01):
        super().__init__()
        self.fusion1 = DepthConvBlock(ch3, ch3, inplace=inplace)
        self.fusion2 = DepthConvBlock(ch2, ch2, inplace=inplace)
        self.fusion3 = DepthConvBlock(ch1, ch1, inplace=inplace)

        self.pix_w_maker = DepthConvBlock(ch3*3, ch3, stride=1, inplace=inplace)
        self.pix_up1 = DepthConv(ch3, ch3, depth_kernel=3, stride=1, inplace=inplace)
        self.pix_up2 = DepthConv(ch3, ch3, depth_kernel=3, stride=1, inplace=inplace)
        self.pix_up3 = subpel_conv3x3(ch3, ch2, 2)
        self.adp1 = DepthConvBlock(ch3, ch3*3, inplace=inplace)
        self.adp2 = nn.Sequential(
            subpel_conv3x3(ch3*2, ch2*3, 2),
        )
        self.adp3 = nn.Sequential(
            subpel_conv3x3(ch2*2, ch1*3, 2),
        )
        self.pix_to_feat = nn.Sequential(
            DepthConv(3, ch3, depth_kernel=3, stride=2, inplace=inplace),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            DepthConv(ch3, ch3, depth_kernel=3, stride=2, inplace=inplace),
        )
        self.feat_up = nn.Sequential(
            subpel_conv3x3(g_ch_16x, ch3, 2),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            subpel_conv3x3(ch3, ch3, 2),
        )
    
    def forward(self, feature, warp_frame_b, warp_frame_f, context1_b, context2_b, context3_b, context1_f, context2_f, context3_f):
        warp_frame_b = self.pix_to_feat(warp_frame_b)
        warp_frame_f = self.pix_to_feat(warp_frame_f)
        feature = self.feat_up(feature)

        pix_w = self.pix_w_maker(torch.cat([feature, warp_frame_b, warp_frame_f], dim=1))
        pix_w1 = self.pix_up1(pix_w)
        out1 = self.adp1(pix_w1)
        scale1_b, scale1_f, bias1 = torch.chunk(out1, 3, dim=1)
        out1 = (1+scale1_b)*context3_b + (1+scale1_f)*context3_f + bias1
        out1 = self.fusion1(out1)

        pix_w2 = self.pix_up2(pix_w)
        out2 = self.adp2(torch.cat([pix_w2, out1], dim=1))
        scale2_b, scale2_f, bias2 = torch.chunk(out2, 3, dim=1)
        out2 = (1+scale2_b)*context2_b + (1+scale2_f)*context2_f + bias2
        out2 = self.fusion2(out2)

        pix_w3 = self.pix_up3(pix_w)
        out3 = self.adp3(torch.cat([pix_w3, out2], dim=1))
        scale3_b, scale3_f, bias3 = torch.chunk(out3, 3, dim=1)
        out3 = (1+scale3_b)*context1_b + (1+scale3_f)*context1_f + bias3
        out3 = self.fusion3(out3)
        return out3, out2, out1



class MvEnc_multi(nn.Module):
    class Multi_flow_fusion(nn.Module):
        def __init__(self, input_channel=2, channel=64, inplace=False, slope=0.01):
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Conv2d(input_channel, channel, 3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=slope, inplace=inplace),
                nn.Conv2d(channel, channel, 1),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(channel + input_channel, channel, 3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=slope, inplace=inplace),
                nn.Conv2d(channel, channel, 1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(channel + input_channel, channel, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=slope, inplace=inplace),
                nn.Conv2d(channel, channel, 1),
            )

        def forward(self, flow0, flow1, flow2):
            x = self.conv0(flow0)
            x = self.conv1(torch.cat([x, flow1], dim=1))
            x = self.conv2(torch.cat([x, flow2], dim=1))
            return x

    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(channel*3, channel, stride=2, inplace=inplace),
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
        )
        self.enc_2 = DepthConvBlock(channel, channel, inplace=inplace)
        self.fusion = self.Multi_flow_fusion(input_channel, channel, inplace=inplace)
        self.dpb_adapter = Layer_type_adapter_Base(channel, channel, channel, inplace=inplace)
        self.dpb_conv = nn.Sequential(
            DepthConvBlock(channel*2, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, stride=2, inplace=inplace),
        )
        self.ref_fusion = DepthConvBlock(channel*3, channel, inplace=inplace), #没用到么？

    def forward(self, flows_f, flows_b, quant_step, mv_ref_info, dpb):
        flow_f = self.fusion(flows_f[0], flows_f[1], flows_f[2])
        flow_b = self.fusion(flows_b[0], flows_b[1], flows_b[2])

        out = self.enc_1(torch.cat([flow_f, flow_b, mv_ref_info], dim=1))
        out = out * quant_step

        feat_f = dpb['dpb_f']['ref_mv_feature']
        if feat_f is not None:
            feat_f = self.dpb_conv(feat_f)
        feat_b = dpb['dpb_b']['ref_mv_feature']
        if feat_b is not None:
            feat_b = self.dpb_conv(feat_b)
        
        out = self.dpb_adapter(out, feat_f, feat_b)
        out = self.enc_2(out)
        return out
    
class MvDec_multi(nn.Module):
    def __init__(self, channel, out_channel=2, inplace=False):
        super().__init__()
        self.dec_0 = nn.Sequential(
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            ResidualBlockUpsample(channel, channel*2, 2, inplace=inplace),
        )
        self.adapter_0 = nn.Conv2d(channel, out_channel, 3, stride=1, padding=1)
        self.adapter_1 = nn.Conv2d(channel, out_channel, 3, stride=1, padding=1)
        self.adapter_2 = nn.Conv2d(channel, out_channel, 3, stride=1, padding=1)
        self.ref_fusion = DepthConvBlock(channel*3, channel*2, inplace=inplace)

    def forward_single(self, out, quant_step):
        out = out * quant_step
        flow2_f = self.adapter_2(out)
        out = self.dec_1(out)
        flow1_f = self.adapter_1(out)
        out = self.dec_0(out)
        flow0_f = self.adapter_0(out)
        return [flow0_f, flow1_f, flow2_f]
    
    def forward(self, x, mv_ref_info, quant_step):
        feature = self.dec_2(x)
        feature = self.ref_fusion(torch.cat((feature, mv_ref_info), dim=1))
        out_f, out_b = torch.chunk(feature, 2, dim=1)
        flows_f = self.forward_single(out_f, quant_step)
        flows_b = self.forward_single(out_b, quant_step)
        return flows_f, flows_b, feature


class F_B_ContextFusion(nn.Module):
    def __init__(self, ch1, ch2, ch3, inplace=False):
        super().__init__()
        self.fusion1 = DepthConvBlock(ch1*2, ch1, inplace=inplace)
        self.fusion2 = DepthConvBlock(ch2*2, ch2, inplace=inplace)
        self.fusion3 = DepthConvBlock(ch3*2, ch3, inplace=inplace)
    
    def forward(self, context1_b, context2_b, context3_b, context1_f, context2_f, context3_f, ):
        out1 = self.fusion1(torch.cat([context1_b, context1_f], dim=1))
        out2 = self.fusion2(torch.cat([context2_b, context2_f], dim=1))
        out3 = self.fusion3(torch.cat([context3_b, context3_f], dim=1))
        return out1, out2, out3

class Layer_type_adapter_Base(nn.Module):
    def __init__(self, ch_x, ch_y, ch_out, inplace=False):
        super().__init__()
        self.adpater_ii = DepthConvBlock(ch_x, ch_out, inplace=inplace)
        self.adpater_ib = DepthConvBlock(ch_x + ch_y, ch_out, inplace=inplace)
        self.adpater_bi = DepthConvBlock(ch_x + ch_y, ch_out, inplace=inplace)
        self.adpater_bb = DepthConvBlock(ch_x + ch_y * 2, ch_out, inplace=inplace)
    
    def forward(self, x, y1, y2):
        inputs = [x]
        if y1 is not None: inputs.append(y1)
        if y2 is not None: inputs.append(y2)
        
        x = torch.cat(inputs, dim=1)
        
        return {
            'ii': self.adpater_ii, 
            'ib': self.adpater_ib, 
            'bi': self.adpater_bi, 
            'bb': self.adpater_bb
        }[('i' if y1 is None else 'b') + ('i' if y2 is None else 'b')](x)








class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=True, max_layer=5):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        channel_N = 64

        self.optic_flow = ME_Spynet()
        self.align = OffsetDiversity(inplace=inplace, aux_feature_num=g_ch_1x+3*2+2*3)

        self.mv_encoder = MvEnc_multi(2, channel_mv)
        self.mv_decoder = MvDec_multi(channel_mv, 2)
        self.mv_ref_extracter = nn.Sequential(
            nn.Conv2d(2*2, g_ch_1x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_1x, channel_mv, 3, stride=2, padding=1),
        )

        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N, inplace=inplace)


        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 2, inplace=inplace),
        )

        

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.Conv2d(g_ch_1x, g_ch_1x, 1)
        self.layer_feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(max_layer)])

        self.feature_extractor = FeatureExtractor(inplace=inplace)
        self.context_fusion_net = MultiScaleContextFusion(inplace=inplace)

        # Line 534-535 is modified by Kuaishou

        self.mv_y_prior_fusion_LTA = Layer_type_adapter_Base(channel_mv, channel_mv, ch_out=channel_mv*2, inplace=inplace)
        self.y_prior_fusion_adaptor_LTA = Layer_type_adapter_Base(g_ch_16x * 2, g_ch_16x, ch_out=g_ch_16x*3, inplace=inplace)
        
        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
        )
        # Line 542-550 is modified by Kuaishou
        self.mv_temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(2*2, g_ch_1x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_1x, channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )



        self.contextual_encoder = ContextualEncoder(inplace=inplace)

        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(g_ch_16x, g_ch_16x, True, inplace=inplace)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )


        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace)
        self.recon_generation_net = ReconGeneration(inplace=inplace)

        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None

        # Line 599-601 is modified by Kuaishou
        self.fb_context_fusion_Enc = FB_ContextFusion_mix_Enc(g_ch_1x, g_ch_2x, g_ch_4x, inplace=inplace)
        self.fb_context_fusion_Dec = FB_ContextFusion_mix_Dec(g_ch_1x, g_ch_2x, g_ch_4x, inplace=inplace)
        self.context3_fusion = DepthConvBlock(g_ch_4x*2, g_ch_4x, inplace=inplace)


    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            mv_y_q_scale_enc_fine = np.linspace(np.log(self.mv_y_q_scale_enc[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_enc[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_enc_fine = np.exp(mv_y_q_scale_enc_fine)
            mv_y_q_scale_dec_fine = np.linspace(np.log(self.mv_y_q_scale_dec[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_dec[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_dec_fine = np.exp(mv_y_q_scale_dec_fine)

            y_q_scale_enc_fine = np.linspace(np.log(self.y_q_scale_enc[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_enc[3, 0, 0, 0]), 64)
            self.y_q_scale_enc_fine = np.exp(y_q_scale_enc_fine)
            y_q_scale_dec_fine = np.linspace(np.log(self.y_q_scale_dec[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_dec[3, 0, 0, 0]), 64)
            self.y_q_scale_dec_fine = np.exp(y_q_scale_dec_fine)

    def bulid_dpb_item(self,):
        dpb_dict = {
            "ref_frame": None,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }
        return dpb_dict

    # Line 632-695 is modified by Kuaishou
    def multi_level_flow(self, x1, x2, flownet):
        def cal_flow_pad(frame1, frame2, padding=16):
            pic_height, pic_width = frame1.shape[-2:]
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, padding)
            frame1_pad = F.pad(frame1, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            frame2_pad = F.pad(frame2, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            flow_l1 = flownet(frame1_pad, frame2_pad)
            flow_l1 = F.pad(flow_l1, (-padding_l, -padding_r, -padding_t, -padding_b))
            return flow_l1
        
        flow_l0 = flownet(x1, x2)
        x1_2x = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2_2x = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        flow_l1 = cal_flow_pad(x1_2x, x2_2x)
        x1_4x = F.interpolate(x1_2x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2_4x = F.interpolate(x2_2x, scale_factor=0.5, mode='bilinear', align_corners=False)

        flow_l2 = cal_flow_pad(x1_4x, x2_4x)
        return [flow_l0, flow_l1, flow_l2], [x1, x1_2x, x1_4x], [x2, x2_2x, x2_4x]

    def multi_level_warp(self, xs, flows):
        xs_warp = []
        for i, (x, flow) in enumerate(zip(xs, flows)):
            x_warp = flow_warp(x, flow)
            xs_warp.append(x_warp)
        return xs_warp

    def res_prior_param_decoder(self, z_hat, dpb, context3, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y_f = dpb["dpb_f"]["ref_y"]
        ref_y_b = dpb["dpb_b"]["ref_y"]
        params = torch.cat([temporal_params, hierarchical_params], dim=1)
        params = self.y_prior_fusion_adaptor_LTA(params, ref_y_f, ref_y_b)

        params = self.y_prior_fusion(params)
        return params


    def multi_scale_feature_extractor(self, dpb, layer_index):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor(dpb["ref_feature"])
        
        feature = self.layer_feature_adaptor[layer_index](feature)
        return self.feature_extractor(feature)


    def motion_compensation(self, dpb, mvs, warpframe, ref_mvs, layer_index):
        ref_frame = dpb["ref_frame"]
        mv1, mv2, mv3 = mvs
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, layer_index)
        context1_init = flow_warp(ref_feature1, mv1)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv1, ref_frame, ref_mvs), dim=1), mv1)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)

        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3


    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        mv_y_q_scale_enc = ckpt["mv_y_q_scale_enc"].reshape(-1)
        mv_y_q_scale_dec = ckpt["mv_y_q_scale_dec"].reshape(-1)
        return y_q_scale_enc, y_q_scale_dec, mv_y_q_scale_enc, mv_y_q_scale_dec

    # Line 708-717 is modified by Kuaishou
    def mv_prior_param_decoder(self, mv_z_hat, dpb, est_mv_bf, est_mv_fb, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        ref_mv_y_f = dpb["dpb_f"]["ref_mv_y"]
        ref_mv_y_b = dpb["dpb_b"]["ref_mv_y"]

        mv_params = self.mv_y_prior_fusion_LTA(mv_params, ref_mv_y_f, ref_mv_y_b)
        mv_temporal_params = self.mv_temporal_prior_encoder(torch.cat((est_mv_bf, est_mv_fb), dim=1))
        mv_params = self.mv_y_prior_fusion(torch.cat((mv_params, mv_temporal_params), dim=1))
        return mv_params




    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def motion_estimation_and_mv_encoding(self, x, dpb, mv_y_q_enc):
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        ref_mv_feature = dpb["ref_mv_feature"]
        mv_y = self.mv_encoder(est_mv, ref_mv_feature, mv_y_q_enc)
        return mv_y

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, self.mv_y_q_basic_enc, q_index=q_index)
        mv_y_q_scale_dec = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, self.mv_y_q_basic_dec, q_index=q_index)

        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec

    def compress(self, x, dpb, q_in_ckpt, q_index, layer_index):
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        # Line 749-755 is modified by Kuaishou
        flows_f, _, ref_frames_f = self.multi_level_flow(x, dpb["dpb_f"]["ref_frame"], self.optic_flow)
        flows_b, xs, ref_frames_b = self.multi_level_flow(x, dpb["dpb_b"]["ref_frame"], self.optic_flow)
        est_mv_fb = self.optic_flow(dpb["dpb_b"]["ref_frame"], dpb["dpb_f"]["ref_frame"]) * 0.5  
        est_mv_bf = self.optic_flow(dpb["dpb_f"]["ref_frame"], dpb["dpb_b"]["ref_frame"]) * 0.5 
        
        mv_ref_info = self.mv_ref_extracter(torch.cat((est_mv_fb, est_mv_bf), dim=1))
        mv_y = self.mv_encoder(flows_f, flows_b, mv_y_q_enc, mv_ref_info, dpb)

        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = torch.round(mv_z)
        # Line 761 is modified by Kuaishou
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, est_mv_bf, est_mv_fb, slice_shape)

        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
            mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = \
            self.compress_four_part_prior(
                mv_y, mv_params,
                self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
                self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        # Line 771-781 is modified by Kuaishou
        mvs_hat_f, mvs_hat_b, mv_feature = self.mv_decoder(mv_y_hat, mv_ref_info, mv_y_q_dec)

        warp_frame_f = flow_warp(ref_frames_f[0], mvs_hat_f[0])
        warp_frame_b = flow_warp(ref_frames_b[0], mvs_hat_b[0])

        context1_f, context2_f, context3_f = self.motion_compensation(dpb["dpb_f"], mvs_hat_f, warp_frame_f, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)
        context1_b, context2_b, context3_b = self.motion_compensation(dpb["dpb_b"], mvs_hat_b, warp_frame_b, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)

        context1, context2, context3 = self.fb_context_fusion_Enc(x, warp_frame_b, warp_frame_f, 
                                                                  context1_b, context2_b, context3_b, 
                                                                  context1_f, context2_f, context3_f)
        
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = torch.round(z)
        # Line 790 is modified by Kuaishou
        context3_prior = self.context3_fusion(torch.cat([context3_f, context3_b], dim=1))
        params = self.res_prior_param_decoder(z_hat, dpb, context3_prior, slice_shape)

        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z_mv.encode(mv_z_hat)
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        self.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        self.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()
        # Line 812-814 is modified by Kuaishou
        context1_hat, context2_hat, context3_hat = self.fb_context_fusion_Dec(y_hat, warp_frame_b, warp_frame_f, 
                                                                  context1_b, context2_b, context3_b, 
                                                                  context1_f, context2_f, context3_f)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1_hat, context2_hat, context3_hat, y_q_dec)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result
    
    def decompress(self, dpb, string, height, width, q_in_ckpt, q_index, layer_index):
        _, mv_y_q_dec, _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(string)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        # Line 842-844 is modified by Kuaishou
        est_mv_fb = self.optic_flow(dpb["dpb_b"]["ref_frame"], dpb["dpb_f"]["ref_frame"]) * 0.5  
        est_mv_bf = self.optic_flow(dpb["dpb_f"]["ref_frame"], dpb["dpb_b"]["ref_frame"]) * 0.5   
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, est_mv_bf, est_mv_fb, slice_shape)
        mv_y_hat = self.decompress_four_part_prior(mv_params,
                                                   self.mv_y_spatial_prior_adaptor_1,
                                                   self.mv_y_spatial_prior_adaptor_2,
                                                   self.mv_y_spatial_prior_adaptor_3,
                                                   self.mv_y_spatial_prior)
        # Line 851-862 is modified by Kuaishou
        mv_ref_info = self.mv_ref_extracter(torch.cat((est_mv_fb, est_mv_bf), dim=1))

        mvs_hat_f, mvs_hat_b, mv_feature = self.mv_decoder(mv_y_hat, mv_ref_info, mv_y_q_dec)
        warp_frame_f = flow_warp(dpb["dpb_f"]["ref_frame"], mvs_hat_f[0])
        warp_frame_b = flow_warp(dpb["dpb_b"]["ref_frame"], mvs_hat_b[0])

        context1_f, context2_f, context3_f = self.motion_compensation(dpb["dpb_f"], mvs_hat_f, warp_frame_f, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)
        context1_b, context2_b, context3_b = self.motion_compensation(dpb["dpb_b"], mvs_hat_b, warp_frame_b, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)

        context3_prior = self.context3_fusion(torch.cat([context3_f, context3_b], dim=1))
        params = self.res_prior_param_decoder(z_hat, dpb, context3_prior, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        # Line 870-872 is modified by Kuaishou
        context1_hat, context2_hat, context3_hat = self.fb_context_fusion_Dec(y_hat, warp_frame_b, warp_frame_f, 
                                                                  context1_b, context2_b, context3_b, 
                                                                  context1_f, context2_f, context3_f)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1_hat, context2_hat, context3_hat, y_q_dec)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
        }


    def encode_decode(self, x, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, layer_index=None):
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            # Line 893-894 is modified by Kuaishou
            encoded = self.compress(x, dpb, q_in_ckpt, q_index, layer_index)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, layer_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, layer_index, string = decode_p(output_path)
            # Line 900-901 is modified by Kuaishou
            decoded = self.decompress(dpb, string, pic_height, pic_width,
                                      q_in_ckpt, q_index, layer_index)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result
        # Line 912-913 is modified by Kuaishou
        encoded = self.forward_one_frame(x, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index,
                                         layer_index=layer_index)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "bit_mv": encoded['bit_mv_y'] + encoded['bit_mv_z'],
            "bit_y_z": encoded['bit_y'] + encoded['bit_z'],
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result


    def forward_one_frame(self, x, dpb, q_in_ckpt=False, q_index=None, layer_index=None):
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        # Line 928-934 is modified by Kuaishou
        flows_f, _, ref_frames_f = self.multi_level_flow(x, dpb["dpb_f"]["ref_frame"], self.optic_flow)
        flows_b, xs, ref_frames_b = self.multi_level_flow(x, dpb["dpb_b"]["ref_frame"], self.optic_flow)
        est_mv_fb = self.optic_flow(dpb["dpb_b"]["ref_frame"], dpb["dpb_f"]["ref_frame"]) * 0.5   
        est_mv_bf = self.optic_flow(dpb["dpb_f"]["ref_frame"], dpb["dpb_b"]["ref_frame"]) * 0.5   
       
        mv_ref_info = self.mv_ref_extracter(torch.cat((est_mv_fb, est_mv_bf), dim=1))
        mv_y = self.mv_encoder(flows_f, flows_b, mv_y_q_enc, mv_ref_info, dpb)

        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = self.quant(mv_z)
        # Line 940 is modified by Kuaishou
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, est_mv_bf, est_mv_fb, slice_shape)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mvs_hat_f, mvs_hat_b, mv_feature = self.mv_decoder(mv_y_hat, mv_ref_info, mv_y_q_dec)
        # Line 947-960 is modified by Kuaishou
        warp_frames_f = self.multi_level_warp(ref_frames_f, mvs_hat_f)
        warp_frames_b = self.multi_level_warp(ref_frames_b, mvs_hat_b)

        warp_frame_f = warp_frames_f[0]
        warp_frame_b = warp_frames_b[0]

        context1_f, context2_f, context3_f = self.motion_compensation(dpb["dpb_f"], mvs_hat_f, warp_frame_f, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)
        context1_b, context2_b, context3_b = self.motion_compensation(dpb["dpb_b"], mvs_hat_b, warp_frame_b, 
                                                                      torch.cat((est_mv_fb, est_mv_bf), dim=1), layer_index)

        context1, context2, context3 = self.fb_context_fusion_Enc(x, warp_frame_b, warp_frame_f, 
                                                                  context1_b, context2_b, context3_b, 
                                                                  context1_f, context2_f, context3_f)
        
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        # Line 966 is modified by Kuaishou
        context3_prior = self.context3_fusion(torch.cat([context3_f, context3_b], dim=1))
        params = self.res_prior_param_decoder(z_hat, dpb, context3_prior, slice_shape)
        
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        # Line 974-976 is modified by Kuaishou
        context1_hat, context2_hat, context3_hat = self.fb_context_fusion_Dec(y_hat, warp_frame_b, warp_frame_f, 
                                                                  context1_b, context2_b, context3_b, 
                                                                  context1_f, context2_f, context3_f)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1_hat, context2_hat, context3_hat, y_q_dec)


        B, _, H, W = x.size()
        pixel_num = H * W
      
        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat
        
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_mv": bpp_mv_y + bpp_mv_z,
                "bpp_y_z": bpp_y + bpp_z ,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
               
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_mv_feature": mv_feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }

