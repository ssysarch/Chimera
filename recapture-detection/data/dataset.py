#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import json
import os
import random

from config import load_config
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils import transforms



class RecaptureDataset(Dataset):
    def __init__(self, phase="train", cfg=None):
        super().__init__()
        self.data = []
        self.label = []
        self.phase = phase
        self.cfg = cfg if cfg is not None else load_config("twob_dct")

        if phase == "test":
            self.populate_data(cfg.test_raw_dirnames, cfg.test_recap_dirnames, True)
        else:
            self.populate_data(cfg.train_raw_dirnames, cfg.train_recap_dirnames)

        # shuffle data
        random.seed(42)
        c = list(zip(self.data, self.label))
        random.shuffle(c)
        self.data, self.label = zip(*c)

        if phase == "train":
            self.data = self.data[: int(len(self.data) * 0.8)]
            self.label = self.label[: int(len(self.label) * 0.8)]
        elif phase == "val":
            self.data = self.data[int(len(self.data) * 0.8) :]
            self.label = self.label[int(len(self.label) * 0.8) :]

        assert len(self.data) == len(self.label)

        if phase == "train":
            self.transform = self._training_transform()
        else:
            self.transform = self._evaluation_transform()

        if cfg.model == "Res50TBNet":
            self.freq_transform = self._freq_transform()

    def blur_transform(self, image, level):
        img_size = image.size[0]
        radius = level * img_size * 0.002
        new_img = image.filter(ImageFilter.GaussianBlur(radius))
        return new_img
        # adjust saturation
        # enhancer = ImageEnhance.Color(image)
        # return enhancer.enhance(level)

    def populate_data(self, raw_dirnames, recap_dirnames, four_classes=False):
        def get_all_files(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    yield os.path.join(root, file)
        for dirname in raw_dirnames + recap_dirnames:
            label = 0 if dirname in raw_dirnames else 1
            for file in get_all_files(dirname):
                if file[-3:].lower() == "png" or file[-3:].lower() == "jpg":
                    # if label == 0 and "fake" in file and self.phase != "test":
                    #     continue
                    if four_classes:
                        if "fake" in file:
                            self.data.append(file)
                            self.label.append(label + 2)
                        else:
                            self.data.append(file)
                            self.label.append(label)
                    else:
                        self.data.append(file)
                        self.label.append(label)

        print(f"Total data: {len(self.data)}")
        print(f"Data with label 0: {self.label.count(0)}")
        print(f"Data with label 1: {self.label.count(1)}")
        print(f"Data with label 2: {self.label.count(2)}")
        print(f"Data with label 3: {self.label.count(3)}")
        # exit()
        # for dirname in recap_dirnames:
        #     for file in get_all_files(dirname):
        #         if file[-3:].lower() == "png" or file[-3:].lower() == "jpg":
        #             self.data.append(file)
        #             self.label.append(1)

    def __getitem__(self, idx):
        true_idx = idx
        target = self.label[true_idx]
        sample = Image.open(self.data[true_idx]).convert("RGB")

        sample_img = self.transform(sample)
        if self.cfg.blur and random.random() < 0.8 and self.phase == "train":
            kernel_size = sample_img.size(1) // 30
            if kernel_size % 2 == 0:
                kernel_size += 1
            blur_transf = T.GaussianBlur(kernel_size, sigma=(0.1, 8.0))
            sample_img = blur_transf(sample_img)

        # # save tensor as image for debugging
        # sample.save(f"sample_{idx}.png")

        if self.cfg.model == "Res50TBNet":
            if self.cfg.blur and self.phase == "train":
                sample = T.ToPILImage()(sample_img)
            sample_dct = self.freq_transform(sample)
        else:
            sample_dct = 0

        return sample_dct, sample_img, target, self.data[true_idx]

    def __len__(self):
        # fact = 5 if (cfg.blur and self.phase != "test") else 1
        # return len(self.data) * fact
        return len(self.data)

    def _training_transform(self):
        transf = T.Compose(
            [
                T.Resize(self.cfg.img_size),
                T.RandomHorizontalFlip(p=self.cfg.h_flip_p),
                T.RandomVerticalFlip(p=self.cfg.v_flip_p),
                T.ToTensor(),
            ]
        )

        if self.cfg.normalize:
            transf.transforms.append(T.Normalize(mean=self.cfg.data_mean, std=self.cfg.data_std))

        return transf

    def _evaluation_transform(self):
        transf = T.Compose(
            [
                T.Resize(self.cfg.img_size),
                T.ToTensor(),
            ]
        )
        if self.cfg.normalize:
            transf.transforms.append(T.Normalize(mean=self.cfg.data_mean, std=self.cfg.data_std))

        return transf

    def _freq_transform(self):
        if self.cfg.transform == "DWT":
            return T.Compose(
                [T.Resize(self.cfg.img_size), transforms.DWT(), T.Resize(self.cfg.img_size)]
            )
        elif self.cfg.transform == "DCT":
            return T.Compose([T.Resize(self.cfg.img_size), transforms.DCT()])
        elif self.cfg.transform == "FFT":
            return T.Compose([T.Resize(self.cfg.img_size), transforms.FFT()])
        elif self.cfg.transform == "Laplacian":
            return T.Compose([T.Resize(self.cfg.img_size), transforms.Laplacian()])
        else:
            raise NotImplementedError(f"Transform {self.cfg.transform} not implemented")
