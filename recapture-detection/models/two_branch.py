#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class Res50TBNet(nn.Module):
    def __init__(self, compress_factor=0.25):
        super(Res50TBNet, self).__init__()
        self.compress_factor = compress_factor
        self.conv1x1_dct1 = None
        self.conv1x1_dct2 = None
        self.conv1x1_rgb1 = None
        self.conv1x1_rgb2 = None
        if compress_factor < 1.0:
            self.conv1x1_dct1 = nn.Conv2d(
                256, int(256 * self.compress_factor), kernel_size=1, stride=1
            )
            self.conv1x1_dct2 = nn.Conv2d(
                1024, int(1024 * self.compress_factor), kernel_size=1, stride=1
            )
            self.conv1x1_dct3 = nn.Conv2d(
                2048, int(2048 * self.compress_factor), kernel_size=1, stride=1
            )
            self.conv1x1_rgb1 = nn.Conv2d(
                256, int(256 * self.compress_factor), kernel_size=1, stride=1
            )
            self.conv1x1_rgb2 = nn.Conv2d(
                1024, int(1024 * self.compress_factor), kernel_size=1, stride=1
            )
            self.conv1x1_rgb3 = nn.Conv2d(
                2048, int(2048 * self.compress_factor), kernel_size=1, stride=1
            )
        self.dct_branch = ResNet50Branch()
        self.rgb_branch = ResNet50Branch()
        self.ca_l1 = CrossAttention(dim=int(256 * self.compress_factor))
        self.ca_l3 = CrossAttention(dim=int(1024 * self.compress_factor))
        self.ca_l4 = CrossAttention(dim=int(2048 * self.compress_factor))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(int(6656 * self.compress_factor))
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(in_dim=int(6656 * self.compress_factor))

    def forward(self, dct_img, rgb_img):
        dct_l1, dct_l3, dct_l4 = self.dct_branch(dct_img)
        rgb_l1, rgb_l3, rgb_l4 = self.rgb_branch(rgb_img)

        if self.compress_factor < 1.0:
            dct_l1 = self.conv1x1_dct1(dct_l1)
            dct_l3 = self.conv1x1_dct2(dct_l3)
            dct_l4 = self.conv1x1_dct3(dct_l4)
            rgb_l1 = self.conv1x1_rgb1(rgb_l1)
            rgb_l3 = self.conv1x1_rgb2(rgb_l3)
            rgb_l4 = self.conv1x1_rgb3(rgb_l4)

        a1 = self.ca_l1(dct_l1, rgb_l1)
        a2 = self.ca_l3(dct_l3, rgb_l3)
        a3 = self.ca_l4(dct_l4, rgb_l4)
        a = torch.concat((a1, a2, a3), dim=1)
        a = self.bn1(a)
        a = self.relu(a)
        a = self.avg_pool(a)
        a = a.flatten(1)
        a = self.mlp(a)
        return a


class ResNet50Branch(nn.Module):
    def __init__(self):
        super(ResNet50Branch, self).__init__()
        model = resnet50(weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        x = self.layer2(l1)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        return l1, l3, l4


class CrossAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False, output_size=(7, 7)):
        super(CrossAttention, self).__init__()
        self.output_size = output_size
        # scale factor
        self.scale = dim**-0.5
        # qkv for branch1
        self.wq1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv1 = nn.Linear(dim, dim, bias=qkv_bias)
        # qkv for branch2
        self.wq2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv2 = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(-2, -1)  # [N,C,H,W] -> [N,HW,C]
        y = y.flatten(2).transpose(-2, -1)  # [N,C,H,W] -> [N,HW,C]
        # q, k, v for branch1
        q1 = self.wq1(x)  # [N,HW,C] -> [N,HW,C]
        k1 = self.wk1(x)  # [N,HW,C] -> [N,HW,C]
        v1 = self.wv1(x)  # [N,HW,C] -> [N,HW,C]
        # q, k , v for branch2
        q2 = self.wq2(y)  # [N,HW,C] -> [N,HW,C]
        k2 = self.wk2(y)  # [N,HW,C] -> [N,HW,C]
        v2 = self.wv2(y)  # [N,HW,C] -> [N,HW,C]
        # get cross attention output of branch1
        attn1 = (
            q1 @ k2.transpose(-2, -1) * self.scale
        )  # [N,HW,C] @ [N,C,HW] -> [N, HW, HW]
        attn1 = attn1.softmax(dim=-1)
        x_a = (attn1 @ v2) + x  # [N,HW,HW] @ [N,HW,C] -> [N,HW,C]
        x_a = x_a.transpose(-2, -1).reshape(
            N, C, H, W
        )  # [N,HW,C] -> [N,C,HW] -> [N,C,H,W]
        # get cross attention output of branch2
        attn2 = (
            q2 @ k1.transpose(-2, -1) * self.scale
        )  # [N,HW,C] @ [N,C,HW] -> [N, HW, HW]
        attn2 = attn2.softmax(dim=-1)
        y_a = (attn2 @ v1) + y  # [N,HW,HW] @ [N,HW,C] -> [N,HW, C]
        y_a = y_a.transpose(-2, -1).reshape(
            N, C, H, W
        )  # [N,HW,C] -> [N,C,HW] -> [N,C,H,W]
        # concat cross attention from branch1 and branch2
        z = torch.concat((x_a, y_a), dim=1)  # [N,C,H,W] -> [N,2C,H,W]
        z = nn.functional.interpolate(
            input=z, size=self.output_size, mode="bilinear"
        )  # [N,C,7,7]
        return z


class MLP(nn.Module):
    def __init__(self, in_dim=6656):
        super(MLP, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(in_dim, 1024),
        #     nn.Linear(1024, 2)
        # )
        self.model = nn.Linear(in_dim, 2)

    def forward(self, x):
        x = self.model(x)
        return x
