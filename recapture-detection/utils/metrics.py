#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import torch
import torch.nn as nn


class MetricAccuracy(nn.Module):
    def __init__(self):
        super(MetricAccuracy, self).__init__()
        self.argmax = torch.argmax
        self.eq = torch.eq
        self.mean = torch.mean

    def forward(self, pred_score, target):
        pred = self.argmax(pred_score, dim=1).to(target.dtype)
        corr = self.eq(pred, target).to(torch.float32)
        acc = self.mean(corr)
        return acc


class PNCounter(nn.Module):
    def __init__(self):
        super(PNCounter, self).__init__()
        self.argmax = torch.argmax
        self.sum = torch.sum
        self.eq = torch.eq
        self.mean = torch.mean

    def forward(self, pred_score, target):
        pred = self.argmax(pred_score, dim=1).to(target.dtype)
        tp = self.sum((pred != 0) * (target != 0))
        fp = self.sum((pred != 0) * (target == 0))
        tn = self.sum((pred == 0) * (target == 0))
        fn = self.sum((pred == 0) * (target != 0))
        return tp, fp, tn, fn


class FourClassCounter(nn.Module):
    def __init__(self):
        super(FourClassCounter, self).__init__()
        self.argmax = torch.argmax
        self.sum = torch.sum
        self.eq = torch.eq
        self.mean = torch.mean

    def forward(self, pred_score, target):
        pred = self.argmax(pred_score, dim=1).to(target.dtype)
        # target 0 and 2 == pred should be 0
        zero_acc = self.sum((pred == 0) & (target == 0))
        two_acc = self.sum((pred == 0) & (target == 2))
        one_acc = self.sum((pred == 1) & (target == 1))
        three_acc = self.sum((pred == 1) & (target == 3))
        return zero_acc, one_acc, two_acc, three_acc
