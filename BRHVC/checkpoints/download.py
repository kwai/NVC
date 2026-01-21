import urllib.request


def download_one(url, target):
    urllib.request.urlretrieve(url, target)


def main():
    urls = {
        'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211259&authkey=AO_gFvTcYZUFd9U': 'cvpr2023_image_psnr.pth.tar',
    }
    for url in urls:
        target = urls[url]
        print("downloading", target)
        download_one(url, target)
        print("downloaded", target)


if __name__ == "__main__":
    main()
