#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Dialated CRF
: Validation for GANet
: Author - Xi Mo
: Institute - University of Kansas
: Date - 4/26/2021
: Last Update - 7/10/2021
: License: Apache 2.0
"""

import numpy as np
import math
import torch
from pathlib import Path
from utils.configuration import CONFIG

''' 
    Options for save metrics to disk
    
    "all":          Write all metrics
    "iou":          Jaccard index per cls
    "acc":          correct pixels per cls
    "dice":         F1-score per cls
    "precision":    Precisoin per cls
    "recall":       Recall per cls
    "roc":          TPR and FPR for calculating ROC per cls
    "mcc":          Phi coefficient per cls
    
'''

# Metric for Offline/Online CPU mode
class Metrics:
    def __init__(self, label, gt, one_hot=False):
        assert gt.ndim == 2, "groundtruth must be grayscale image"
        # one_hot label to unary
        if one_hot:
            assert label.ndim == 3, "label must be 3-dimensional for one-hot encoding"
            label = np.argmax(label, axis = 2)
        else:
            assert label.ndim == 2, "label must be 2-dimensional"

        self.H, self.W = CONFIG["SIZE"]
        label_area, gt_area = np.where(label == CONFIG["NUM_CLS"] - 1), \
                              np.where(gt == CONFIG["NUM_CLS"] - 1)
        self.label_area = set(zip(label_area[0], label_area[1]))
        self.gt_area = set(zip(gt_area[0], gt_area[1]))
        self.TP_FN = len(self.gt_area)
        self.TP_FP = len(self.label_area)
        self.TP = len(self.label_area.intersection(self.gt_area))
        self.FP = self.TP_FP - self.TP
        self.TN = self.H * self.W - self.TP_FN - self.TP_FP + self.TP
        self.FN = self.TP_FN - self.TP

    # Jaccard: TP/(FP+TP+FN)
    def IOU(self) -> np.float32:
        UN = self.TP_FN + self.TP_FP
        if UN == 0: return 1.0
        return np.float32(self.TP / (UN - self.TP + 1e-31))

    # acc: TP+TN/(TP+FP+TN+FN)
    def ACC(self) -> np.float32:
        accuracy = (self.TP + self.TN) / (self.H * self.W)
        return np.float32(accuracy)

    # dice: Sørensen–Dice coefficient 1/(1/precision + 1/recall)
    # precision, recall(aka TPR), FPR
    def DICE(self) -> [np.float32]:
        if self.TP == self.FN == self.FP == 0: return 1.0
        precision = np.float32(self.TP / (self.TP + self.FP + 1e-31))
        recall = np.float32(self.TP / (self.TP + self.FN + 1e-31))
        dice = np.float32(2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-31))
        return dice

    # precision
    def PRECISION(self) -> np.float32:
        if self.TP == self.FN == self.FP == 0: return 1.0
        return np.float32(self.TP / (self.TP + self.FP + 1e-31))

    # recall
    def RECALL(self) -> np.float32:
        if self.TP == self.FN == self.FP == 0: return 1.0
        return np.float32(self.TP / (self.TP + self.FN + 1e-31))

    # TPR and FPR for ROC curve
    def ROC(self) -> [np.float32]:
        if self.TP == self.FN == 0 and self.FP == self.TN == 0:
            return [1.0, 1.0]

        if not (self.TP == self.FN == 0) and self.FP == self.TN == 0:
            tpr = np.float32(self.TP / (self.TP + self.FN))
            return [tpr, 1.0]

        if not (self.FP == self.TN == 0) and self.TP == self.FN == 0:
            fpr = np.float32(self.FP / (self.FP + self.TN))
            return [1.0, fpr]

        tpr = np.float32(self.TP / (self.TP + self.FN))
        fpr = np.float32(self.FP / (self.FP + self.TN))
        return [tpr, fpr]

    # mcc: Matthews correlation coefficient (Phi coefficient)
    def MCC(self) -> np.float32:
        if self.TP == self.FN == self.FP == 0: return 1.0
        N = self.TN + self.TP + self.FN + self.FP
        S = (self.TP + self.FN) / N
        P = (self.TP + self.FP) / N
        if S == 0 or P == 0: return -1.0
        if S == 1 or P == 1: return 0.0
        return np.float32((self.TP / N - S * P) / math.sqrt(P * S * (1-S) * (1-P)))

    # evalute and save results to disk
    '''
        options:
        'all': save all evalutaion metrics to disk
        otherwise: specify the metric to be saved, refer to line 19
    '''
    def save_to_disk(self, name: str, path: Path, option="all"):
        path = path.joinpath("evaluation.txt")

        if option == "all":
            with open(path, "a+") as f:
                iou, acc, dice = 100 * self.IOU(), 100 * self.ACC(), 100 * self.DICE()
                precsion, recall = 100 * self.PRECISION(), 100 * self.RECALL()
                tpr, fpr = self.ROC()
                tpr *= 100
                fpr *= 100
                mcc = 100 * self.MCC()
                f.write(f"{name} iou:{iou:.2f} acc:{acc:.2f} precision:{precsion:.2f} "
                        f"recall:{recall:.2f} dice:{dice:.2f} "
                        f"tpr:{tpr:.2f} fpr:{fpr:.2f} mcc:{mcc:.2f}\n")
            return
        # write iou only
        if option == "iou":
            with open(path, "a+") as f:
                iou = 100 * self.IOU()
                f.write(f"{name:s} iou:{iou:.2f}\n")
            return
        # write acc only
        if option == "acc":
            with open(path, "a+") as f:
                acc = 100 * self.self.ACC()
                f.write(f"{name:s} acc:{acc:.2f}\n")
            return
        # write dice only
        if option == "dice":
            with open(path, "a+") as f:
                dice = 100 * self.DICE()
                f.write(f"{name:s} dice:{dice:.2f}\n")
            return
        # write precision only
        if option == "precision":
            with open(path, "a+") as f:
                precision = 100 * self.PRECISION()
                f.write(f"{name:s} precision:{precision:.2f}\n")
            return
        # write recall only
        if option == "recall":
            with open(path, "a+") as f:
                recall = 100 * self.RECALL()
                f.write(f"{name:s} precision:{recall:.2f}\n")
            return
        # write roc only
        if option == "roc":
            with open(path, "a+") as f:
                tpr, fpr = 100 * self.ROC()
                f.write(f"{name:s} tpr:{tpr:.2f} fpr:{fpr:.2f}\n")
            return
        # write mcc only
        if option == "mcc":
            with open(path, "a+") as f:
                mcc = 100 * self.MCC()
                f.write(f"{name:s} mcc:{mcc:.2f}\n")
            return

    # generate evaluations on-the-fly
    '''
    
        Return a dict of metrics for further processing, options:
        "all" or []:    all metrics
        [metrics]:      selected metrics by names, refer to line 19('roc' -> 'tpr' and 'fpr')
        
    '''
    def values(self, options="all"):
        varDict = {"iou":None, "acc": None, "dice":None, "precision": None,"recall":None,
                   "tpr": None, "fpr":None, "mcc": None}
        if options == "all" or options == []:
            options = ["iou", "acc", "dice", "precision", "recall", "tpr", "fpr", "mcc"]

        for metric in options:
            if metric == "iou": varDict[metric] = self.IOU()
            if metric == "acc": varDict[metric] = self.ACC()
            if metric == "dice": varDict[metric] = self.DICE()
            if metric == "precision": varDict[metric] = self.PRECISION()
            if metric == "recall": varDict[metric] = self.RECALL()
            if metric == "tpr": varDict[metric] = self.ROC()[0]
            if metric == "fpr": varDict[metric] = self.ROC()[1]
            if metric == "mcc": varDict[metric] = self.MCC()

        return varDict