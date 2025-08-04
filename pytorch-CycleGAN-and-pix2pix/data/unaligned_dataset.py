import os
import random

from PIL import Image, ImageOps

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(
        #     opt.dataroot, opt.phase + "A"
        # )  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(
        #     opt.dataroot, opt.phase + "B"
        # )  # create a path '/path/to/data/trainB'
        # self.dir_A = "/home/seongbin/recap-det/data/CNN_synth_testset/stylegan2"
        # self.dir_A = "/home/seongbin/recap-det/data/progan_val/raw"
        self.dir_A = "/home/seongbin/recap-det/data2/fakeface"
        self.dir_B = "/home/seongbin/recap-det/data2/fakeface"
        # self.dir_B = "/home/seongbin/recap-det/data/iphone/stylegan2F"
        # self.dir_B = "/home/seongbin/recap-det/data/iphone/stylegan2D"
        # self.dir_B = "/home/seongbin/recap-det/data/progan_val/recapture"
        # self.dir_B = "/home/seongbin/recap-det/data/sashas_cam/stylegan2_blur0"
        # self.dir_B = "/home/seongbin/recap-det/data/sashas_cam/stylegan2_moire_1"
        # self.dir_B = "/home/seongbin/recap-det/data/sashas_cam/stylegan2_blur_2"

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        print(len(self.A_paths), len(self.B_paths))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = self.transform_A

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # assert A_path.split("/")[-1] == B_path.split("/")[-1]
        A_img = Image.open(A_path).convert("RGB")
        # add padding around the image
        # padding = 40
        # A_img = ImageOps.expand(A_img, padding, fill="white")

        # A_img = A_img[3:256, 0:255]
        # if "horse" in A_path:
        #     A_img = A_img.crop((0, 0, 253, 256))
        # else:
        # A_img = A_img.crop((1, 1, 256, 256))
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # # save overlay of a and b for debugging
        # from torchvision import transforms as tvtr

        # to_pil = tvtr.ToPILImage()
        # # denormalize
        # A = A * 0.5 + 0.5
        # B = B * 0.5 + 0.5
        # A_img_pil = to_pil(A)
        # B_img_pil = to_pil(B)
        # overlay = Image.blend(A_img_pil, B_img_pil, 0.5)
        # overlay.save(f"/home/seongbin/recap-det/data/hi.png")
        # exit()

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
