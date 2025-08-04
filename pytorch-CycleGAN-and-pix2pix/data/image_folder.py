"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import json
import os

import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# def make_dataset(dir, max_dataset_size=float("inf")):
#     folder = "/home/seongbin/recap-det/pytorch-CycleGAN-and-pix2pix/results/attack_bas2/test_25/images/cat/0_real"
#     images = []
#     for img in os.listdir(folder):
#         if is_image_file(img):
#             print(img)
#             images.append(os.path.join(folder, img))

#     return images[: min(max_dataset_size, len(images))]


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    # Add images from the JSON file
    # json_file_path = "/home/seongbin/recap-det/data/mytrain.json"
    # json_file_path = "/home/seongbin/recap-det/data/progan_paths.json"
    # json_file_path = "/home/seongbin/recap-det/data/mytest.json"
    # if os.path.isfile(json_file_path):
    #     with open(json_file_path, "r") as f:
    #         json_files = json.load(f)
    #         for file in json_files:
    #             images.append(os.path.join(dir, file))
    print("Total images: ", len(images))
    return images[: min(max_dataset_size, len(images))]
    # return ["/home/seongbin/recap-det/pytorch-CycleGAN-and-pix2pix/checkpoints/attack_bas2/web/images/epoch001_fake_A.png"]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
