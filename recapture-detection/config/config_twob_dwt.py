#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import os

import torch
from utils import logger


class DefaultConfig:
    # ------------------------------------------------------------------
    # model config
    # ------------------------------------------------------------------
    model = "Res50TBNet"  # model name
    train_raw_dirnames = [
    ]
    train_recap_dirnames = [
    ]
    test_raw_dirnames = [
    ]
    test_recap_dirnames = [
    ]

    # 0.867 & 0.915 & 0.952 & 0.973 & 0.418 & 0.335
    # 0.590 & 0.540 & 0.298 & 0.210
    blur = True

    run_name = "twob_dwt"

    train_model_path = None
    resume_epoch = 0
    test_model_path = ""
    output_path = "./output/all_blur"

    compress_factor = 0.25
    Res50TBNet = {"compress_factor": compress_factor}
    # ------------------------------------------------------------------
    # train test dataloader config
    # ------------------------------------------------------------------
    train_batch_size = 8
    val_batch_size = 8
    test_batch_size = 8
    num_workers = 8
    prefetch_factor = 2
    pin_mem = True
    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------
    optimizer = "RAdam"
    Adam = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 1.0e-4,
    }

    SGD = {
        "lr": 1.0e-6,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "dampening": 0,
        "nesterov": False,
    }

    RAdam = {
        "lr": 1.0e-5,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 1.0e-4,
    }

    # ------------------------------------------------------------------
    # loss function
    # ------------------------------------------------------------------
    loss_fn = "CrossEntropyLoss"
    CrossEntropyLoss = {
        "weight": None,
        "size_average": None,
        "reduction": "mean",
        "label_smoothing": 0.1,
    }
    # ------------------------------------------------------------------
    # lr scheduler
    # ------------------------------------------------------------------
    scheduler = "CosineAnnealingWarmupLR"
    max_epoch = 100
    CosineAnnealingWarmupLR = {
        "warmup_iters": 5,
        "max_epochs": max_epoch,
        "lr_max": 0.1,
        "lr_min": 1e-6,
    }

    OneCycleLR = {
        "max_lr": 0.01,
        "total_steps": 160 * 5,
        "epochs": None,
        "steps_per_epoch": None,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "div_factor": 1000.0,  # initial_lr = max_lr/div_factor
        "final_div_factor": 1e4,  # min_lr = initial_lr/final_div_factor
        "three_phase": False,
    }

    # ------------------------------------------------------------------
    # img transforms
    # ------------------------------------------------------------------
    img_size = (256, 256)
    h_flip_p = 0.5
    v_flip_p = 0.5
    data_mean = [0.6235, 0.6006, 0.5880]
    data_std = [0.2236, 0.2346, 0.2490]
    split = False
    include_print = False
    normalize = False
    transform = "DWT"
    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    device = torch.device("cuda:1")
    save_model = True

    def parse(self):
        logger.info("User Configuration: ")
        print("=" * 50)
        with open(
            os.path.join(os.path.join(self.output_path, self.run_name), "config.txt"),
            "w",
        ) as f:
            for k, v in self.__class__.__dict__.items():
                if not k.startswith("_") and not isinstance(v, dict) and k != "parse":
                    section = {
                        "model": "Model Configuration",
                        "train_batch_size": "Dataloader Configuration",
                        "optimizer": "Optimizer",
                        "scheduler": "Scheduler",
                        "img_size": "Image Transformation",
                        "device": "Device Configuration",
                    }
                    if k in section:
                        print("-" * 50)
                        f.write(("-" * 50) + "\n")
                        print(section[k])
                        f.write(section[k] + "\n")
                        print("-" * 50)
                        f.write(("-" * 50) + "\n")
                    print(k, getattr(self, k))
                    f.write(f"{k} {getattr(self, k)}\n")
                    if k == "optimizer" or k == "scheduler" or k == "loss_fn":
                        print(getattr(self, v))
                        f.write(f"{getattr(self, v)}\n")
        print("=" * 50)
