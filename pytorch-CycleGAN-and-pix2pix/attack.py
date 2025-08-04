"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os

import cv2
import numpy as np
import torch
import torchattacks
import torchvision.utils as vutils
from torch import nn
from torchvision import transforms

from data import create_dataset
from models import Res50TBNet, create_model
from options.test_options import TestOptions
from util import html, util


class MyModelWrapper(nn.Module):
    def __init__(self, gan, model, device):
        super(MyModelWrapper, self).__init__()
        self.dct_transform = DCT().to(device)
        self.gan = gan
        self.model = model.to(device)
        self.device = device

    def forward(self, x):
        x = self.gan.netG(x)
        # x is tensor, change to array of PIL images
        transformed_x = [self.dct_transform(transforms.ToPILImage()(img)) for img in x]
        transformed_x = torch.stack(transformed_x).to(self.device)
        return self.model(x, transformed_x)


class DCT(torch.nn.Module):
    def __init__(self, k=10):
        super(DCT, self).__init__()
        self.k = k

    def forward(self, pil_img):
        return self._filter_bank_preprocess(pil_img)

    @staticmethod
    def _filter_bank_preprocess(pil_img, k: int = 10):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        img_dct = cv2.dct(img)
        img_dct_f = np.abs(img_dct)

        low_freq = img_dct_f > 2 * k
        mid_freq = (img_dct_f >= k) * (img_dct_f < 2 * k)
        high_freq = img_dct_f < k

        img_dct_low = cv2.idct(img_dct * low_freq)[None, :, :]
        img_dct_mid = cv2.idct(img_dct * mid_freq)[None, :, :]
        img_dct_high = cv2.idct(img_dct * high_freq)[None, :, :]
        return (
            torch.from_numpy(
                np.concatenate((img_dct_low, img_dct_mid, img_dct_high), axis=0)
            ).contiguous()
            / 255.0
        )


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    gan = create_model(opt)  # create a model given opt.model and other options
    gan.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    classifier = Res50TBNet(compress_factor=0.25)
    model_path = "/home/seongbin/recap-det/Two-branch-Document-Recapture/output/iphone/twob_dct_ft1/models/12_0.928.pt"
    classifier.load_state_dict(torch.load(model_path)["model"])
    model = MyModelWrapper(gan, classifier, gan.device)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    attk = torchattacks.PGD(model, eps=16 / 255)
    attk.set_normalization_used((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        labels = torch.ones(data["A"].shape[0], dtype=torch.long).to(gan.device)
        save_path = data["A_paths"][0].replace(
            "/home/seongbin/recap-det/data/CNN_synth_testset/stylegan2",
            webpage.get_image_dir(),
        )
        data = data["A"].to(gan.device)
        # apply attack
        im_data = attk(data, labels)
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # # util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        im_data = (im_data + 1) / 2
        vutils.save_image(im_data, save_path)

    webpage.save()  # save the HTML
