#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Dialted CRF
: Network frameworks and helpers
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/24/2021
: Last Update - 7/10/2021
: License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as ops
import time
import math

from pathlib import Path
from utils.configuration import CONFIG


class DSConv(nn.Module):
    def __init__(self, depth=CONFIG["NUM_CLS"], stride=1):
        super(DSConv, self).__init__()
        self.dsconv = nn.Sequential(
            nn.Conv2d(1, 1, (depth, 1), stride = stride),
            nn.BatchNorm2d(1, affine = True),
            nn.ReLU6(inplace = True)
            )

    def forward(self, feat):
        convFeat = self.dsconv(feat)
        avgFeat = ops.adaptive_avg_pool3d(feat, (1, 1, 1))
        maxFeat = ops.adaptive_max_pool3d(feat, (1, 1, 1))
        return avgFeat, maxFeat


class Aggregate(nn.Module):
    def __init__(self, inChan = CONFIG["NUM_CLS"], scale = 0.25):
        super(Aggregate, self).__init__()
        self.DSCModules = nn.ModuleList([])
        H, W = CONFIG["SIZE"]
        H, W = int(H * scale), int(W * scale)
        d = int(math.sqrt(H * W)/2) # for full-size, use d = int(math.sqrt(H * W) / 10)
        # d = int(math.sqrt(H * W) / 10)
        # upsampling for average global pooling
        self.upSampleAvg = nn.Sequential(
            nn.Linear(d, H * W, bias = True),
            nn.LayerNorm(H * W,),
            nn.ReLU6(inplace = True)
            )
        # upsampling for max global pooling
        self.upSampleMax = nn.Sequential(
            nn.Linear(d, H * W, bias = True),
            nn.LayerNorm(H * W),
            nn.ReLU6(inplace = True)
            )

        for _ in range(d):
            self.DSCModules.append(DSConv(inChan))

    def forward(self, feat):
        for sk, module in enumerate(self.DSCModules):
            if sk == 0:
                avgFeat, maxFeat = module(feat)
            else:
                tmpAvgFeat, tmpMaxFeat = module(feat)
                avgFeat = torch.cat((avgFeat, tmpAvgFeat), 3)
                maxFeat = torch.cat((maxFeat, tmpMaxFeat), 3)

        avgFeat = self.upSampleAvg(avgFeat)
        maxFeat = self.upSampleMax(maxFeat)
        return avgFeat, maxFeat


class global_energy(nn.Module):
    def __init__(self, inChan = CONFIG["NUM_CLS"], scale = 0.125):
        super(global_energy, self).__init__()
        self.inChan = inChan
        self.scale = scale
        H, W = CONFIG["SIZE"]
        self.H, self.W = int(H * scale), int(W * scale)
        self.globalFeats = Aggregate(inChan, self.scale)
        self.conv = nn.Sequential(
            nn.Conv2d(2 + CONFIG["NUM_CLS"], CONFIG["NUM_CLS"], 1, bias = True),
            nn.BatchNorm2d(CONFIG["NUM_CLS"], affine = True),
            nn.ReLU6(inplace = True)
            )

    def forward(self, feat):
        transFeat = feat.view(feat.shape[0], feat.shape[1], 1, -1)
        transFeat = torch.transpose(transFeat, 1, 2)
        avgFeat, maxFeat = self.globalFeats(transFeat)
        avgFeat = avgFeat.view(avgFeat.shape[0], 1, self.H, self.W)
        maxFeat = maxFeat.view(maxFeat.shape[0], 1, self.H, self.W)
        Feat = self.conv(torch.cat((avgFeat, maxFeat, feat), 1))
        return Feat


class dialated_crf(nn.Module):
    def __init__(self, inChan = CONFIG["NUM_CLS"], scale = 0.125):
        super(dialated_crf, self).__init__()
        self.scale = scale
        if scale != 1:
            self.downSample = nn.FractionalMaxPool2d(3, output_ratio = scale)

        self.getGlobalEnergy = global_energy(inChan = inChan, scale = scale)
        self.getUnary = nn.Sequential(
            nn.Conv2d(CONFIG["NUM_CLS"], CONFIG["NUM_CLS"], 1, bias = False),
            nn.BatchNorm2d(CONFIG["NUM_CLS"]),
            nn.ReLU6(inplace = True)
            )
        self.normlizer = nn.Sequential(
            nn.Conv2d(2 * inChan, inChan, 1, 1, bias=False),
            nn.BatchNorm2d(inChan)
            )

    def forward(self, feat):
        if self.scale != 1: feat = self.downSample(feat)
        unary = self.getUnary(feat)
        out = self.getGlobalEnergy(feat)
        out = torch.cat((out, unary), dim = 1)
        out = self.normlizer(out)
        if self.scale != 1: out = ops.interpolate(out, scale_factor=int(1 / self.scale))
        return out

# optimizer parser
def optimizer(_net):
    if CONFIG["OPTIM"] == "adamw":
        optimizer = torch.optim.AdamW(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"],
                                         amsgrad      = CONFIG["AMSGRAD"])
    elif CONFIG["OPTIM"] == "adam":
        optimizer = torch.optim.Adam(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"],
                                         amsgrad      = CONFIG["AMSGRAD"])
    elif CONFIG["OPTIM"] == "sgd":
        optimizer = torch.optim.SGD(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         momentum     = CONFIG["MOMENT"],
                                         weight_decay = CONFIG["DECAY"],
                                         dampening    = CONFIG["DAMPEN"],
                                         nesterov     = CONFIG["NESTROV"])
    elif CONFIG["OPTIM"] == "rmsprop":
        optimizer = torch.optim.RMSprop(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         momentum     = CONFIG["MOMENT"],
                                         weight_decay = CONFIG["DECAY"],
                                         alpha        = CONFIG["ALPHA"],
                                         eps          = CONFIG["EPS"],
                                         centered     = CONFIG["CENTERED"])
    elif CONFIG["OPTIM"] == "rprop":
        optimizer = torch.optim.Rprop(_net.parameters(),
                                         lr         = CONFIG["LR"],
                                         etas       = CONFIG["ETAS"],
                                         step_sizes = CONFIG["STEPSIZE"])
    elif CONFIG["OPTIM"] == "adagrad":
        optimizer = torch.optim.Adagrad(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         lr_decay     = CONFIG["LR_DECAY"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "adadelta":
        optimizer = torch.optim.Adadelta(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         rho          = CONFIG["RHO"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "sparseadam":
        optimizer = torch.optim.SparseAdam(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "adamax":
        optimizer = torch.optim.Adamax(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "asgd":
        optimizer = torch.optim.ASGD(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         lambd        = CONFIG["LAMBD"],
                                         alpha        = CONFIG["ALPHA"],
                                         weight_decay = CONFIG["DECAY"],
                                         t0           = CONFIG["T0"])
    else:
        raise NameError(f"Unsupported optimizer {CONFIG['OPTIM']}, please customize it.")

    return optimizer

# Write model to disk
def save_model(baseDir: Path, network: torch.nn.Module, epoch: int, logger: {},
               optimizer: torch.optim, postfix="dcrf"):
    date = time.strftime(f"%Y%m%d-%H%M%S-Epoch-{epoch}_{postfix}.pt", time.localtime())
    path = baseDir.joinpath(date)
    print("\nNow saveing model to:\n%s" %path)
    torch.save({
        'epoch': epoch,
        'logs': logger,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    print("Done!\n")