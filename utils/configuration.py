#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Dialated CRF
: Configuraion profile
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/25/2021
: Last Update - 7/10/2021
: License: Apache 2.0
"""

import argparse
import math
from pathlib import Path


parser = argparse.ArgumentParser(description="Arguments for training, validation and testing")

# Training
parser.add_argument("-r", "--restore", action = "store_true",
                    help = "Restore training by loading specified checkpoint or lattest checkpoint")
parser.add_argument("-train", action = "store_true",
                    help = r"Train network, trainset requires to be saved to disk beforehand")

# Accomodation to suction dataset
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by users")
parser.add_argument("-d", "--dir", type = Path, default = r"dataset",
                    help = r"The main folder to read/save training/validation/test samples")

# Validation
parser.add_argument("-v", "--validate", action = "store_true",
                    help = "Validate results using metrics")

CONFIG = {

    "SCALE": 0.5,              # select a dowm-sampling scale from [0.125, 0.25, 0.5, 1] etc.
    "POSTFIX": ".png",         # label/sample image postfix to read or save as
    "SIZE": (480, 640),        # input size specification: (H, W)

    "HAS_NORM": False,                           # normailzationm,for samples only
    "PAR_NORM": {"mean": (0.485, 0.456, 0.406),  # dictionary format with tuples
                 "std": (0.229, 0.224, 0.225)},  # valid for train and test

    # Training

    # Loss function
    "LOSS": "ce",               # choose loss function between, "ce", "bce", "huber", "poisson"
    "WEIGHT": [0.25, 0.25, 0.5],    # size = NUM_CLS, set None to disable, for "ce",
    # bce losses
    "BETA": 0.618,              # beta for huber loss
    "FULL": True,               # add the Stirling approximation term to Poisson loss
    "PEPS": 1e-8,               # eps for Poisson loss
    "REDUCT": "mean",           # loss reduction method, "mean", "sum" , "none"

    # optimizer
    "OPTIM": "adamw",           # "sgd", "adam", "adamw", "rmsprop", "rprop", "adagrad", "adadelta"
                                # and "sparseadam", "adamax", "asgd"
    "LR": 0.001,                # learning rate
    "BETAS": (0.9, 0.999),      # coefficients for computing running averages of gradient and its square
    "EPS": 1e-08,               # term added to the denominator to improve numerical stability
    "DECAY": 0,                 # weight decay (L2 penalty)
    "AMSGRAD": True,            # use the AMSGrad variant
    "MOMENT": 0,                # momentum factor
    "DAMPEN": 0,                # dampening for momentum
    "NESTROV": False,           # enables Nesterov momentum
    "ALPHA": 0.99,              # smoothing constant
    "CENTERED": False,          # gradient is normalized by estimation of variance
    "ETAS": (0.5, 1.2),         # multiplicative increase and decrease factors (etaminus, etaplis)
    "STEPSIZE": (1e-06, 50),    # minimal and maximal allowed step sizes
    "LR_DECAY": 0,              # learning rate decay
    "RHO": 0.9,                 # coefficient for computing a running average of squared gradients
    "LAMBD": 1e-4,              # decay term
    "T0": 1e6,                  # point at which to start averaging

    # training
    "BATCHSIZE": 120,           # batchsize for training
    "EPOCHS": 1000,             # epoches for training
    "SHOW_LOSS": 10,            # number of minibatchs processed to print training info
    "SAVE_MODEL": 10,           # epoch intervel to save, start counting from epoch 2
    "SHUFFLE": True,            # random shuffle
    "NUM_WORKERS": 0,           # set to 0 if memorry error occurs
    "PIN_MEMORY": True,         # set to false if memory is insufficient
    "DROP_LAST": False,
    "NUM_CLS": 3,               # number of classes
    "INT_CLS": (255, 0, 128),   # raw label intensity levels to differentiate classes in training

    # Validation
    "VAL_INT": (0, 128, 255),   # Intensity levels for DCRF groundtruth during validation
    "PED_INT": (0, 128, 255),   # Intensity levels for DCRF predictions during validation

    # Augmentation
    "HOR_FLIP": True,          # random horizontal flip
    "PH_FLIP": 0.5,            # must be a number in [0, 1]

    "VER_FLIP": True,          # random vertical flip
    "PV_FLIP": 0.5,            # must be a number in [0, 1]

    "SHIFT": True,             # random affine transform, will not affect the label
    "PAR_SFT": (0.2, 0.2),     # must be a tuple if set "SHIFT" to True
    "P_SFT": 0.6,              # probablity to shift

    "ROTATE": True,            # rotate image
    "ROT_DEG": math.pi,        # rotation degree
    "P_ROT": 0.4,              # probability to rotate

    "COLOR_JITTER": True,           # random color random/fixed jitter
    "P_JITTER": 0.2,                # probability to jitter
    "BRIGHTNESS": 0.5,              # random brightness adjustment, float or (float, float)
    "CONTRAST": 0.5,                # random brightness adjustment, float or (float, float)
    "SATURATION": 0.5,              # random saturation adjustment, float or (float, float)
    "HUE": 0.25,                    # random hue adjustment, float or (float, float)

    "BLUR": True,                   # random gaussian blur
    "P_BLUR": 0.3,                  # probability to blur image
    "PAR_BLUR":
        {"kernel": 15,              # kernal size, can be either one int or [int, int]
         "sigma": (0.5, 3.0)}       # sigma, can be single one float
    }