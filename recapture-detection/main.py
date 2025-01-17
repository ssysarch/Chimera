#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import utils.schedulers as schedulers
from config import load_config

# from config import cfg
from data.dataset import RecaptureDataset
from utils import cal_acc, cal_pn, cal_test, logger


def train(cfg):
    setup_seed(42)
    # make dirs
    if not os.path.exists(cfg.output_path):
        os.mkdir(cfg.output_path)
    run_out_path = os.path.join(cfg.output_path, cfg.run_name)
    if not os.path.exists(run_out_path):
        os.mkdir(run_out_path)
    model_out_path = os.path.join(run_out_path, "models")
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    cfg.parse()

    # data preparation
    train_dataset = RecaptureDataset(phase="train", cfg=cfg)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_mem,
    )
    val_dataset = RecaptureDataset(phase="val", cfg=cfg)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_mem,
    )

    # model
    # model = getattr(models, cfg.model)(**cfg.Res50TBNet)
    model = getattr(models, cfg.model)()
    model.to(device=cfg.device)
    logger.info("Model Summary: ")
    print(model)

    if cfg.train_model_path:
        saved = torch.load(cfg.train_model_path, map_location="cpu")
        model.load_state_dict(saved["model"] if "model" in saved else saved)
        if cfg.resume_epoch > 0:
            opt = saved["opt"]
            scheduler = saved["scheduler"]
        elif cfg.model == "ViTB16":
            opt = getattr(optim, cfg.optimizer)(
                params=model.model.heads.parameters(), **getattr(cfg, cfg.optimizer)
            )
            scheduler = None
        else:
            opt = getattr(optim, cfg.optimizer)(
                params=model.parameters(), **getattr(cfg, cfg.optimizer)
            )
            scheduler = None
    else:
        # optimizer
        opt = getattr(optim, cfg.optimizer)(
            params=model.parameters(), **getattr(cfg, cfg.optimizer)
        )
        if cfg.scheduler == "None":
            scheduler = None
        elif cfg.scheduler == "CosineAnnealingWarmupLR":
            scheduler = getattr(schedulers, cfg.scheduler)(
                optimizer=opt, **getattr(cfg, cfg.scheduler)
            )
        else:
            scheduler = getattr(optim.lr_scheduler, cfg.scheduler)(
                optimizer=opt, **getattr(cfg, cfg.scheduler)
            )

    # loss function
    loss_fn = getattr(nn, cfg.loss_fn)(**getattr(cfg, cfg.loss_fn))
    loss_fn.to(device=cfg.device)

    # summary writer
    log_dir = os.path.join(run_out_path, "summary")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # training
    train_step = 0
    best_val_acc = 0
    for epoch in range(cfg.resume_epoch, cfg.max_epoch):
        # train model
        model.train()
        train_accs, train_losses = [], []
        TP, FP, TN, FN = 0, 0, 0, 0

        for batch, (dcts, rgbs, ys) in enumerate(tqdm.tqdm(train_loader)):
            # data to device
            dcts = dcts.to(device=cfg.device)
            rgbs = rgbs.to(device=cfg.device)
            ys = ys.to(device=cfg.device)
            # forward pass
            opt.zero_grad()
            logits = model(dcts, rgbs)

            # calculate batch loss, accuracy, tp, fp, tn, fn
            loss = loss_fn(logits, ys)
            acc = cal_acc(logits.detach(), ys)
            tp, fp, tn, fn = cal_pn(logits.detach(), ys)
            TP += tp.item()
            FP += fp.item()
            TN += tn.item()
            FN += fn.item()

            # backward
            loss.backward()
            # keep track of learning rate in iterations
            cur_lr = list(opt.param_groups)[0]["lr"]
            writer.add_scalar("lr", cur_lr, train_step)
            # print log every 500 iters
            if (batch + 1) % 500 == 0:
                logger.debug(
                    f"[train epoch:{epoch}/{cfg.max_epoch}] "
                    f"[batch:{batch}] "
                    f"lr:{cur_lr}|"
                    f"loss:{loss.item():.3f}|"
                    f"accuracy:{acc.item():.3f}"
                )
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            opt.step()
            train_losses.append(loss.item())
            train_accs.append(acc.item())
            train_step += 1

        # print epoch summary
        epoch_acc = np.mean(train_accs)
        epoch_loss = np.mean(train_losses)
        recap_precision = TP / (TP + FP + 1e-8)
        recap_recall = TP / (TP + FN + 1e-8)
        nonrecap_precision = TN / (TN + FN + 1e-8)
        nonrecap_recall = TN / (TN + FP + 1e-8)
        logger.debug(
            f"[train epoch:{epoch}/{cfg.max_epoch} summary:] "
            f"lr:{list(opt.param_groups)[0]['lr']}|"
            f"accuracy:{epoch_acc:.3f}|"
            f"loss:{epoch_loss:.3f}|"
            f"recap precision:{recap_precision:.3f}|"
            f"recap recall:{recap_recall:.3f}|"
            f"non-recap precision:{nonrecap_precision:.3f}|"
            f"non-recap recall:{nonrecap_recall:.3f}"
        )
        writer.add_scalars(
            "train_stats",
            {
                "accuracy": epoch_acc,
                "loss": epoch_loss,
                "recap precision": recap_precision,
                "recap recall": recap_recall,
                "non-recap precision": nonrecap_precision,
                "non-recap recall": nonrecap_recall,
            },
            global_step=epoch,
        )

        # evaluation
        (
            val_acc,
            val_loss,
            recap_precision,
            recap_recall,
            nonrecap_precision,
            nonrecap_recall,
        ) = val2(model, val_loader, loss_fn)

        # print log
        logger.debug(
            f"[val epoch:{epoch}/{cfg.max_epoch} summary:] "
            f"accuracy:{val_acc:.3f}|"
            f"loss:{val_loss:.3f}|"
            f"recap precision:{recap_precision:.3f}|"
            f"recap recall:{recap_recall:.3f}|"
            f"non-recap precision:{nonrecap_precision:.3f}|"
            f"non-recap recall:{nonrecap_recall:.3f}"
        )
        # add to tensorboard
        writer.add_scalars(
            "val_stats",
            {
                "accuracy": val_acc,
                "loss": val_loss,
                "recap precision": recap_precision,
                "recap recall": recap_recall,
                "non-recap precision": nonrecap_precision,
                "non-recap recall": nonrecap_recall,
            },
            global_step=epoch,
        )

        # save model
        if cfg.save_model:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model": model.state_dict(), "opt": opt, "scheduler": scheduler},
                    os.path.join(model_out_path, f"{epoch}_{val_acc:.3f}.pt"),
                )
                logger.info(f"model of epoch {epoch} saved")
            else:
                logger.info(f"model of epoch {epoch} not saved!")

        # update lr
        if cfg.scheduler == "ReduceLROnPlateau":
            scheduler.step(metrics=loss)
        elif scheduler:
            scheduler.step()

    writer.close()


def test(cfg, test_model_path):
    setup_seed(42)
    # model = getattr(models, cfg.model)(**cfg.Res50TBNet)
    model = getattr(models, cfg.model)()
    if test_model_path:
        model.load_state_dict(torch.load(test_model_path, map_location="cpu")["model"])
    else:
        model.load_state_dict(
            torch.load(cfg.test_model_path, map_location="cpu")["model"]
        )
    model.to(device=cfg.device)
    model.eval()

    test_dataset = RecaptureDataset(phase="test", cfg=cfg)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        # prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_mem,
    )

    loss_fn = getattr(nn, cfg.loss_fn)(**getattr(cfg, cfg.loss_fn))
    loss_fn.to(device=cfg.device)

    (
        recap_precision,
        recap_recall,
        nonrecap_precision,
        nonrecap_recall,
    ) = val(model, test_loader, loss_fn)

    logger.debug(
        f"test summary:] "
        # f"accuracy:{test_acc:.3f}|"
        # f"loss:{test_loss:.3f}|"
        f"raw real:{nonrecap_precision:.3f}|"
        f"raw fake:{nonrecap_recall:.3f}|"
        f"recap real:{recap_precision:.3f}|"
        f"recap fake:{recap_recall:.3f}"
    )


def val2(model, val_loader, loss_fn):
    model.eval()
    val_accs, val_losses = [], []
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        for _, (dcts, rgbs, ys) in enumerate(tqdm.tqdm(val_loader)):
            # data to device
            dcts = dcts.to(device=cfg.device)
            rgbs = rgbs.to(device=cfg.device)
            ys = ys.to(device=cfg.device)

            logits = model(dcts, rgbs)
            # calculate batch loss, accuracy, tp, fp, tn, fn
            loss = loss_fn(logits, ys)
            acc = cal_acc(logits, ys)
            tp, fp, tn, fn = cal_pn(logits, ys)
            TP += tp.item()
            FP += fp.item()
            TN += tn.item()
            FN += fn.item()

            val_accs.append(acc.item())
            val_losses.append(loss.item())

    val_acc = np.mean(val_accs)
    val_loss = np.mean(val_losses)
    recap_precision = TP / (TP + FP + 1e-8)
    recap_recall = TP / (TP + FN + 1e-8)
    nonrecap_precision = TN / (TN + FN + 1e-8)
    nonrecap_recall = TN / (TN + FP + 1e-8)

    return (
        val_acc,
        val_loss,
        recap_precision,
        recap_recall,
        nonrecap_precision,
        nonrecap_recall,
    )


def val(model, val_loader, loss_fn):
    false_positive_files = []

    model.eval()
    val_accs, val_losses = [], []
    T0, T1, T2, T3 = 0, 0, 0, 0
    R0, R1, R2, R3 = 0, 0, 0, 0

    with torch.no_grad():
        # for _, (dcts, rgbs, ys) in enumerate(tqdm.tqdm(val_loader)):
        for _, (dcts, rgbs, ys, filen) in enumerate(tqdm.tqdm(val_loader)):
            # data to device
            dcts = dcts.to(device=cfg.device)
            rgbs = rgbs.to(device=cfg.device)
            ys = ys.to(device=cfg.device)

            logits = model(dcts, rgbs)

            pred = torch.argmax(logits, dim=1).to(ys.dtype)
            for i in range(len(pred)):
                if pred[i] == 0 and (ys[i] == 3):
                    false_positive_files.append(filen[i])

            # calculate batch loss, accuracy, tp, fp, tn, fn
            # loss = loss_fn(logits, ys)
            # acc = cal_acc(logits, ys)
            (
                a,
                b,
                c,
                d,
            ) = cal_test(logits, ys)
            R0 += a.item()
            R1 += b.item()
            R2 += c.item()
            R3 += d.item()

            T0 += torch.sum((ys == 0)).item()
            T1 += torch.sum((ys == 1)).item()
            T2 += torch.sum((ys == 2)).item()
            T3 += torch.sum((ys == 3)).item()

            # val_accs.append(acc.item())
            # val_losses.append(loss.item())

    nonrecap_real = R0 / (T0 + 1e-8)
    recap_real = R1 / (T1 + 1e-8)
    nonrecap_fake = R2 / (T2 + 1e-8)
    recap_fake = R3 / (T3 + 1e-8)
    print(R0, T0, R1, T1, R2, T2, R3, T3)

    with open("false_positive_files.txt", "w") as f:
        for file in false_positive_files:
            f.write(file + "\n")

    return (
        recap_real,
        recap_fake,
        nonrecap_real,
        nonrecap_fake,
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", required=False)
    parser.add_argument("--test_path", type=str, default=None, required=False)
    parser.add_argument("--train_all", action="store_true", required=False)
    parser.add_argument("--adv_all", action="store_true", required=False)
    parser.add_argument("--config", type=str, default="twob_dct", required=False)
    parser.add_argument("--test_raw_dirnames", type=str, nargs='*', default=None, required=False)
    parser.add_argument("--test_recap_dirnames", type=str, nargs='*', default=None, required=False)
    args = parser.parse_args()
        
    # import config as cfg
    cfg = load_config(args.config)
    
    if args.test_raw_dirnames is not None:
        cfg.test_raw_dirnames = args.test_raw_dirnames
    if args.test_recap_dirnames is not None:
        cfg.test_recap_dirnames = args.test_recap_dirnames

    if args.test:
        test(cfg, args.test_path)
    else:
        train(cfg)
