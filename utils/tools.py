#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Dialated CRF
: Tool functions
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/25/2021
: Last Update - 7/10/2021
: License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as ops
import time
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image

from utils.configuration import CONFIG

# Helper to read images
def read_image_from_disk(path: Path, isTensor=True, colorImg=True) -> {str: torch.Tensor}:
    img = {}
    for imgFile in sorted(path.rglob("*" + CONFIG["POSTFIX"])):
        if colorImg:
            imgData = Image.open(imgFile)
        else:
            imgData = Image.open(imgFile).convert('L')

        if isTensor:
            img[imgFile.name] = transforms.ToTensor()(imgData)
        else:
            img[imgFile.name] = np.array(imgData)
    return img

# Helper to in-place transforming grayscale labels to classes
def trans_img_to_cls(img: {str:np.ndarray}) -> {str:np.ndarray}:
    for key, im in img.items():
        save = []
        for idx, val in enumerate(CONFIG["VAL_INT"]):
            save.append(np.where(im == val))
        for idx, val in enumerate(save):
            im[val] = idx
        img[key] = im
    return img

# Helper to save images
def save_image_to_disk(img:torch.Tensor, path: Path):
    data = img.cpu().clone()
    if len(data.shape) == 2:
        data = data.to(torch.float32)
        data = data / (CONFIG["NUM_CLS"] - 1 + 1e-31)
        save_image(data, path)