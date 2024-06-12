from __future__ import print_function

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from model.models import MODELS
from road_dataset import DeepGlobeDataset, SpacenetDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util
import torch.multiprocessing as mp

__dataset__ = {"spacenet": SpacenetDataset, "deepglobe": DeepGlobeDataset}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="config file path")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=sorted(MODELS.keys()),
        help="Name of Model = {}".format(MODELS.keys()),
    )
    parser.add_argument("--exp", required=True, type=str, help="Experiment Name/Directory")
    parser.add_argument(
        "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(__dataset__.keys()),
        help="select dataset name from {}. (default: Spacenet)".format(__dataset__.keys()),
    )
    parser.add_argument(
        "--model_kwargs",
        default={},
        type=json.loads,
        help="parameters for the model",
    )
    parser.add_argument(
        "--multi_scale_pred",
        default=True,
        type=util.str2bool,
        help="perform multi-scale prediction (default: True)",
    )
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "AdamW", "SparseAdam", "ASGD", "Adamax", "NAdam", "Rprop"], help="Optimizer choice")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD and RMSprop")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Weight decay")
    parser.add_argument("--betas", default=(0.9, 0.999), type=eval, help="Betas for Adam, AdamW, Adamax, NAdam")
    parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon for optimizers")
    parser.add_argument("--alpha", default=0.99, type=float, help="Alpha for RMSprop and Adadelta")
    parser.add_argument("--rho", default=0.9, type=float, help="Rho for Adadelta")
    return parser.parse_args()

def initialize_experiment(args):
    config = None

    if args.resume is not None:
        if args.config is not None:
            print("Warning: --config overridden by --resume")
            config = torch.load(args.resume)["config"]
    elif args.config is not None:
        config = json.load(open(args.config))

    assert config is not None

    util.setSeed(config)

    experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
    util.ensure_dir(experiment_dir)

    train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
    test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)
    train_loss_file = open(train_file, "w")
    val_loss_file = open(test_file, "w")

    return config, experiment_dir, train_loss_file, val_loss_file

def get_dataloaders(config, args):
    train_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["train_dataset"],
            seed=config["seed"],
            is_train=True,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["train_batch_size"],
        num_workers=0,
        shuffle=True,
        pin_memory=False,
    )

    val_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["val_dataset"],
            seed=config["seed"],
            is_train=False,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["val_batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))
    return train_loader, val_loader

def initialize_model_optimizer(config, args, num_gpus):
    model = MODELS[args.model_name](config["task1_classes"], config["task2_classes"], **args.model_kwargs)
    if num_gpus > 1:
        print(f"Training with multiple GPUs ({num_gpus})")
        model = nn.DataParallel(model)

    optimizer = None
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "SparseAdam":
        optimizer = optim.SparseAdam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps)
    elif args.optimizer == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "Rprop":
        optimizer = optim.Rprop(model.parameters(), lr=args.lr, etas=(args.lr, 1e-6), step_sizes=(args.lr, 1e-6))

    return model, optimizer

def load_checkpoint(args, model, optimizer):
    start_epoch = 1
    best_miou = 0

    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["miou"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        util.weights_init(model, manual_seed=config["seed"])

    viz_util.summary(model, print_arch=False)
    return start_epoch, best_miou

def prepare_losses(config):
    weights = torch.ones(config["task1_classes"])
    if config["task1_weight"] < 1:
        print("Roads are weighted.")
        weights[0] = 1 - config["task1_weight"]
        weights[1] = config["task1_weight"]

    weights_angles = torch.ones(config["task2_classes"])
    if config["task2_weight"] < 1:
        print("Road angles are weighted.")
        weights_angles[-1] = config["task2_weight"]

    angle_loss = CrossEntropyLoss2d(weight=weights_angles, size_average=True, ignore_index=255, reduce=True)
    road_loss = mIoULoss(weight=weights, size_average=True, n_classes=config["task1_classes"])
    return road_loss, angle_loss

def train_epoch(epoch, model, optimizer, train_loader, config, args, train_loss_file, num_gpus, road_loss, angle_loss):
    train_loss_iou, train_loss_vec = 0, 0
    model.train()
    optimizer.zero_grad()
    hist, hist_angles = np.zeros((config["task1_classes"], config["task1_classes"])), np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["train_dataset"][args.dataset]["crop_size"]

    for i, data in enumerate(train_loader, 0):
        inputsBGR, labels, vecmap_angles = data
        inputsBGR = Variable(inputsBGR.float())
        outputs, pred_vecmaps = model(inputsBGR)

        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(labels[0]), False)
            num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0]), False)
            for idx, output in enumerate(outputs):
                hist += util.fast_hist(output.cpu().max(1)[1].data.numpy().flatten(), labels[idx].cpu().numpy().flatten(), config["task1_classes"])
        else:
            loss1 = road_loss(outputs, util.to_variable(labels, True))
            hist += util.fast_hist(outputs.cpu().max(1)[1].data.numpy().flatten(), labels.cpu().numpy().flatten(), config["task1_classes"])

        loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles, True))
        hist_angles += util.fast_hist(pred_vecmaps.cpu().max(1)[1].data.numpy().flatten(), vecmap_angles.cpu().numpy().flatten(), config["task2_classes"])

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        train_loss_iou += loss1.item()
        train_loss_vec += loss2.item()

        if i % 20 == 0:
            print("Epoch: [{}/{}] iter: {} loss: {:.6f}".format(epoch, config["trainer"]["epochs"], i, loss.item()))

    train_epoch_loss = train_loss_iou / len(train_loader)
    train_epoch_loss_vec = train_loss_vec / len(train_loader)
    train_loss_file.write("Epoch: {} \tLoss: {:.6f}\tVec_Loss: {:.6f}\n".format(epoch, train_epoch_loss, train_epoch_loss_vec))
    train_loss_file.flush()

    miou_train = util.compute_score(hist)
    vec_miou_train = util.compute_score(hist_angles)
    print("Train: [{}/{}] road mIoU: {:.6f} vec mIoU: {:.6f}".format(epoch, config["trainer"]["epochs"], miou_train, vec_miou_train))
    return miou_train

def val_epoch(epoch, model, val_loader, config, val_loss_file, road_loss, angle_loss):
    model.eval()
    val_loss_iou, val_loss_vec = 0, 0
    hist, hist_angles = np.zeros((config["task1_classes"], config["task1_classes"])), np.zeros((config["task2_classes"], config["task2_classes"]))

    for i, data in enumerate(val_loader, 0):
        inputsBGR, labels, vecmap_angles = data
        inputsBGR = Variable(inputsBGR.float())
        outputs, pred_vecmaps = model(inputsBGR)

        loss1 = road_loss(outputs, util.to_variable(labels, True))
        hist += util.fast_hist(outputs.cpu().max(1)[1].data.numpy().flatten(), labels.cpu().numpy().flatten(), config["task1_classes"])

        loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles, True))
        hist_angles += util.fast_hist(pred_vecmaps.cpu().max(1)[1].data.numpy().flatten(), vecmap_angles.cpu().numpy().flatten(), config["task2_classes"])

        val_loss_iou += loss1.item()
        val_loss_vec += loss2.item()

    val_epoch_loss = val_loss_iou / len(val_loader)
    val_epoch_loss_vec = val_loss_vec / len(val_loader)
    val_loss_file.write("Epoch: {} \tLoss: {:.6f}\tVec_Loss: {:.6f}\n".format(epoch, val_epoch_loss, val_epoch_loss_vec))
    val_loss_file.flush()

    miou_val = util.compute_score(hist)
    vec_miou_val = util.compute_score(hist_angles)
    print("Val: [{}/{}] road mIoU: {:.6f} vec mIoU: {:.6f}".format(epoch, config["trainer"]["epochs"], miou_val, vec_miou_val))
    return miou_val

def main():
    args = parse_args()

    if args.resume:
        config = torch.load(args.resume)["config"]
    else:
        config = json.load(open(args.config))

    experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
    util.ensure_dir(experiment_dir)

    train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
    test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)
    train_loss_file = open(train_file, "w")
    val_loss_file = open(test_file, "w")

    config, experiment_dir, train_loss_file, val_loss_file = initialize_experiment(args)
    train_loader, val_loader = get_dataloaders(config, args)

    num_gpus = torch.cuda.device_count()
    model, optimizer = initialize_model_optimizer(config, args, num_gpus)
    start_epoch, best_miou = load_checkpoint(args, model, optimizer)

    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    road_loss, angle_loss = prepare_losses(config)

    for epoch in range(start_epoch, config["trainer"]["epochs"] + 1):
        miou_train = train_epoch(epoch, model, optimizer, train_loader, config, args, train_loss_file, num_gpus, road_loss, angle_loss)
        miou_val = val_epoch(epoch, model, val_loader, config, val_loss_file, road_loss, angle_loss)
        scheduler.step()

        is_best = miou_val > best_miou
        best_miou = max(miou_val, best_miou)
        util.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": best_miou,
                "config": config,
            },
            is_best,
            experiment_dir,
        )
        print(f"Best mIoU: {best_miou}")

if __name__ == "__main__":
    main()
