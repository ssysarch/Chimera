# https://github.com/AmadeusITGroup/Moire-Pattern-Detection
# The convolutional layer C1 filter three 512 x 384 input images with 32 kernels of size 7 x 7 with a stride of 1 pixel. The stride of pooling layer S1 is 2 pixels. Then, the convolved images of LH and HL are merged together by taking the maximum from both the images. In the next step, the convolved image of LL is merged with the Max result by multiplying both the results (as explained in section III-B). C2-C4 has 16 kernels of size 3 x 3 with a stride of 1 pixel. S2 pools the merged features with a stride of 4. The dropout is applied to the output of S4 which has been flattened. The fully connected layer FC1 has 32 neurons and FC2 has 1 neuron. The activation of the output layer is a softmax function.

# One significant observation found while analyzing the images is, when the image intensities are relatively darker, the patterns aren’t visible much. The darker regions in the image produce less Moire ́ patterns compared to the brighter regions in the image. To summarize the spread of the Moire ́ pattern in the image, spatially, and to produce this effect while training the network, we used the LL band of the image (which is the downsampled original image consisting of low frequency information) and used it as weights for LH anf HL band during the training, by directly multiplying it to the convolved and combined response of the LH and HL bands

import os

import numpy as np
import torch
from pytorch_wavelets import DWTForward
from torch import nn


class MoireDetNet(nn.Module):
    def __init__(self, depth=3):
        super(MoireDetNet, self).__init__()
        kernel_size_1 = 7  # we will use 7x7 kernels
        kernel_size_2 = 3  # we will use 3x3 kernels
        pool_size = 2  # we will use 2x2 pooling throughout
        conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
        conv_depth_2 = 16  # ...switching to 16 after the first pooling layer
        drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
        drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
        hidden_size = 32  # 128 512 the FC layer will have 512 neurons
        num_classes = 2

        self.dwt = DWTForward(J=1, wave="haar", mode="zero")

        self.conv_1_LL = nn.Conv2d(
            depth, conv_depth_1, kernel_size=kernel_size_1, padding="same"
        )
        self.conv_1_LH = nn.Conv2d(
            depth, conv_depth_1, kernel_size=kernel_size_1, padding="same"
        )
        self.conv_1_HL = nn.Conv2d(
            depth, conv_depth_1, kernel_size=kernel_size_1, padding="same"
        )
        self.conv_1_HH = nn.Conv2d(
            depth, conv_depth_1, kernel_size=kernel_size_1, padding="same"
        )
        self.relu_1_LL = nn.ReLU()
        self.relu_1_LH = nn.ReLU()
        self.relu_1_HL = nn.ReLU()
        self.relu_1_HH = nn.ReLU()
        self.pool_1_LL = nn.MaxPool2d(kernel_size=pool_size)
        self.pool_1_LH = nn.MaxPool2d(kernel_size=pool_size)
        self.pool_1_HL = nn.MaxPool2d(kernel_size=pool_size)
        self.pool_1_HH = nn.MaxPool2d(kernel_size=pool_size)

        self.c4 = nn.Conv2d(
            conv_depth_1, conv_depth_2, kernel_size=kernel_size_2, padding="same"
        )
        self.c4_relu = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.drop_1 = nn.Dropout(drop_prob_1)

        self.c5 = nn.Conv2d(
            conv_depth_2, conv_depth_1, kernel_size=kernel_size_2, padding="same"
        )
        self.c5_relu = nn.ReLU()
        self.s3 = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

        self.c6 = nn.Conv2d(
            conv_depth_1, conv_depth_1, kernel_size=kernel_size_2, padding="same"
        )
        self.c6_relu = nn.ReLU()
        self.s4 = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
        self.drop_2 = nn.Dropout(drop_prob_1)

        self.flat = nn.Flatten()
        # self.hidden = nn.Linear(
        #     conv_depth_1 * 256, hidden_size
        # )  # input dimensions may be wrong
        self.hidden = nn.Linear(
            conv_depth_1 * 16, hidden_size
        )  # input dimensions may be wrong
        self.drop_3 = nn.Dropout(drop_prob_2)
        self.out = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax() # only need if not using CrossEntropyLoss

    def forward(self, _, x):
        LL, arr = self.dwt(x)
        LH, HL, HH = arr[0][:, 0, :, :], arr[0][:, 1, :, :], arr[0][:, 2, :, :]
        x_LL = self.pool_1_LL(self.relu_1_LL(self.conv_1_LL(LL)))
        x_LH = self.pool_1_LH(self.relu_1_LH(self.conv_1_LH(LH)))
        x_HL = self.pool_1_HL(self.relu_1_HL(self.conv_1_HL(HL)))
        x_HH = self.pool_1_HH(self.relu_1_HH(self.conv_1_HH(HH)))

        max_LH_HL_HH = torch.max(torch.max(x_LH, x_HL), x_HH)
        inp_merged = torch.mul(x_LL, max_LH_HL_HH)

        x = self.c4_relu(self.c4(inp_merged))
        x = self.s2(x)
        x = self.drop_1(x)

        x = self.c5_relu(self.c5(x))
        x = self.s3(x)

        x = self.c6_relu(self.c6(x))
        x = self.s4(x)
        x = self.drop_2(x)

        x = self.flat(x)
        x = self.hidden(x)
        x = self.drop_3(x)
        x = self.out(x)

        return x
