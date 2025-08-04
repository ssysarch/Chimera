import torch

from . import networks
from .base_model import BaseModel
from .ffl import FocalFrequencyLoss


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B", "fake_A"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        if opt.simulator:
            self.simulator = networks.define_G(
                opt.input_nc,
                opt.output_nc,
                # opt.ngf,
                64,
                # opt.netG,
                # "unet_128",
                "unet_256",
                opt.norm,
                not opt.no_dropout,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            # freeze the simulator
            for param in self.simulator.parameters():
                param.requires_grad = False
            self.simulator.eval()

        if (
            self.isTrain
        ):  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFreq = FocalFrequencyLoss(loss_weight=opt.freq_weight).to(
                self.device
            )
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            lr_mult = 0.05 if opt.simulator else 1.0
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr * lr_mult, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.simulator:
            self.fake_A = self.netG(self.real_A)
            # sharpen the fake_A
            self.fake_B = self.simulator(self.fake_A)
        else:
            self.fake_A = torch.zeros_like(self.real_A)
            self.fake_B = self.netG(self.real_A)  # G(A)

    def sharpen(self, x):
        # x is a tensor of shape (N, C, H, W)
        kernel = (
            torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).float().to(self.device)
        )
        x = torch.nn.functional.conv2d(x, kernel.unsqueeze(0).unsqueeze(0), padding=1)

        return x

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_A
        fake_AB = torch.cat(
            (self.real_A, self.fake_B), 1
        )  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.opt.simulator:
            real_AB = torch.cat((self.real_A, self.real_A), 1)
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_D_new(self):
        """Calculate GAN loss for the discriminator"""
        # don't use conditional GANs; we only feed the output to the discriminator
        fake_A = self.fake_A
        pred_fake = self.netD(fake_A.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_A = self.real_A
        pred_real = self.netD(real_A)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        if self.opt.simulator:
            self.loss_G_L1 = (
                self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1
            )
            self.loss_Freq = (
                self.criterionFreq(self.fake_B, self.real_A) * self.opt.lambda_L1
            )
        else:
            self.loss_G_L1 = (
                self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            )
            self.loss_Freq = (
                self.criterionFreq(self.fake_B, self.real_B) * self.opt.lambda_L1
            )
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_Freq
        self.loss_G.backward()

    def backward_G_new(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_A = self.fake_A
        pred_fake = self.netD(fake_A)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(
            self.netD, False
        )  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
