
#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import torch
import math


# epoch-based Cosine Annealing with Warmup
class CosineAnnealingWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_iters=10, max_epochs=160, lr_max=0.1, lr_min=1e-5):
        self.warmup_iters = warmup_iters
        self.T_max = max_epochs-self.warmup_iters
        self.lr_max = lr_max
        self.lr_min = lr_min
        super(CosineAnnealingWarmupLR, self).__init__(optimizer=optimizer, lr_lambda=self.cal_lr)

    def cal_lr(self, cur_iter):
        if cur_iter < self.warmup_iters:
            return cur_iter / self.warmup_iters
        else:
            return ((self.lr_min + 0.5*(self.lr_max-self.lr_min) *
                     (1.0+math.cos((cur_iter-self.warmup_iters)
                                   / self.T_max*math.pi)))/0.1)
