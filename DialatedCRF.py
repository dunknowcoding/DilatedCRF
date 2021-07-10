#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Dialated Continuous Random Field
: Training and testing
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/17/2021
: Last Update - 7/10/2021
: License: Apache 2.0
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from pathlib import Path
from matplotlib import pyplot as plt

from utils.configuration import parser, CONFIG
from utils.dataLoader import SuctionGrasping
from utils.validate import Metrics
from utils.tools import read_image_from_disk, save_image_to_disk, trans_img_to_cls
from utils.network import dialated_crf, save_model, optimizer


# Helper for training DCRF
def train_dcrf_label_to_label(_net, _input, _gtLabel, _optimizer, _lossFunc):
    start_time = time.time()
    _optimizer.zero_grad()
    labelOut = _net(_input)
    loss = _lossFunc(labelOut, _gtLabel)
    loss.backward()
    _optimizer.step()
    end_time = time.time()
    runtime = (end_time - start_time) * 1e3
    return labelOut, loss.item(), runtime

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    ''' Train DCRF '''

    if args.train:
        assert 0 < CONFIG["SAVE_MODEL"] <= CONFIG["EPOCHS"], "Invalid interval of screenshot"
        # check image directory to read from
        if str(args.dir) != "dataset":
            if not args.dir.is_dir():
                raise IOError(f"Invalid sample folder:\n{args.dir.resolve()}")

        args.dir = args.dir.joinpath("train")
        outDir, labDir = args.dir.joinpath("output"), args.dir.joinpath("annotations")
        if not outDir.is_dir() or not labDir.is_dir():
            raise IOError(f"Default sample folders not found in:\n{args.dir.resolve()}")
        # create a different ckpt foloder
        ckptDir = Path.cwd().joinpath("checkpoint")
        ckptDir.mkdir(exist_ok=True)
        # load dataset
        trainDCRF_data = SuctionGrasping(outDir, labDir, applyTrans=False)
        # train GCRFNet
        trainDCRF_set = data.DataLoader(dataset      = trainDCRF_data,
                                        batch_size   = CONFIG["BATCHSIZE"],
                                        shuffle      = CONFIG["SHUFFLE"],
                                        num_workers  = CONFIG["NUM_WORKERS"],
                                        pin_memory   = CONFIG["PIN_MEMORY"],
                                        drop_last    = CONFIG["DROP_LAST"])
        # loss function
        if CONFIG["LOSS"] == "ce":  # Cross-Entropy Loss
            if CONFIG["WEIGHT"] is not None:
                weight = torch.FloatTensor(CONFIG["WEIGHT"]).to(device)
            else:
                weight = None

            lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight, reduction=CONFIG["REDUCT"])
        elif CONFIG["LOSS"] == "bce":  # Binary Cross-Entropy Loss
            if CONFIG["WEIGHT"] is not None:
                weight = torch.FloatTensor(CONFIG["WEIGHT"]).to(device)
            else:
                weight = None

            lossFuncLabel = torch.nn.BCELoss(weight=weight, reduction=CONFIG["REDUCT"])
        elif CONFIG["LOSS"] == "huber":  # Huber Loss
            lossFuncLabel = nn.SmoothL1Loss(beta=CONFIG["BETA"], reduction=CONFIG["REDUCT"])
        elif CONFIG["LOSS"] == "poisson":  # Poisson Loss
            lossFuncLabel = nn.PoissonNLLLoss(log_input=False, reduction=CONFIG["REDUCT"],
                                              eps=CONFIG["PEPS"])
        else:
            raise NameError(f"Unspported loss function type '{CONFIG['LOSS']}'.")

        lossFuncLabel = lossFuncLabel.to(device)
        dcrf = dialated_crf(scale = CONFIG["SCALE"])
        optimizer_DCRF = optimizer(dcrf)
        dcrf.train()
        dcrf.to(device)
        # save stats
        logger = {
            "loss": [],
            "precision": [],
            "recall": [],
            "iou": []
        }
        # load checkpoint if restore is true
        if args.restore:
            # checkpoint filepath check
            if str(args.checkpoint) != "checkpoint":
                if not args.checkpoint.is_file():
                    raise IOError(f"Designated checkpoint file does not exist:\n"
                                  f"{args.checkpoint.resolve()}")
                ckptPath = args.checkpoint.resolve
            # Create checkpoint directory
            ckptDir = Path.cwd().joinpath("checkpoint")
            ckptDir.mkdir(exist_ok=True, parents=True)
            # get the lattest checkpoint if set to default directory and restore is true
            if str(args.checkpoint) == "checkpoint" and args.restore:
                fileList = sorted(ckptDir.glob("*.pt"), reverse=True,
                                  key=lambda item: item.stat().st_ctime)
                if len(fileList) == 0:
                    raise IOError(f"Cannot find any checkpoint files in:\n"
                                  f"{ckptDir.resolve()}\n")
                else:
                    ckptPath = fileList[0]

            checkpoint = torch.load(ckptPath)
            print(f"\nCheckpoint loaded:\n{ckptPath}\n")
            dcrf.load_state_dict(checkpoint['model_state_dict'])
            optimizer_DCRF.load_state_dict(checkpoint['optimizer_state_dict'])
            lastEpoch = checkpoint['epoch']
            logger = checkpoint["logs"]
            if lastEpoch == CONFIG["EPOCHS"]:
                print("WARNING: Previous training has been finished, "
                      "initialize transfer training ...... \n")
                lastEpoch = 0
        else:
            lastEpoch = 0

        totalBatch = np.ceil(len(trainDCRF_data) / CONFIG["BATCHSIZE"])
        epochLoss, runPrec, runRec, runIU = 0.0, 0.0, 0.0, 0.0
        for epoch in range(lastEpoch, CONFIG["EPOCHS"]):
            runLoss = 0.0
            for idx, data in enumerate(trainDCRF_set):
                raw = data[0].to(device)
                label = data[1]
                if CONFIG["LOSS"] == "ce":
                    label = label.long()
                elif CONFIG["LOSS"] in ["bce", "huber", "poisson", "kld"]:
                    label = SuctionGrasping.one_hot_encoder(label)

                label = label.to(device)
                raw, loss, runtime = train_dcrf_label_to_label(
                                        dcrf, raw, label, optimizer_DCRF, lossFuncLabel)
                # evaluation of the last batch
                with torch.no_grad():
                    labs, pred = label.detach(), raw.detach()
                    if len(pred.shape) == 4: pred = torch.argmax(pred, dim=1)
                    TP_FP = len(torch.where(pred == CONFIG["NUM_CLS"] - 1)[0])
                    TP_FN = len(torch.where(labs == CONFIG["NUM_CLS"] - 1)[0])
                    TP = len(torch.where(torch.add(pred, labs) == 2 * (CONFIG["NUM_CLS"] - 1))[0])
                    IU = float(torch.div(TP, TP_FP + TP_FN - TP + 1e-31))
                    precision = float(torch.div(TP, TP_FP + 1e-31))
                    recall = float(torch.div(TP, TP_FN + 1e-31))

                runLoss += loss
                epochLoss += loss
                runRec += recall
                runPrec += precision
                runIU += IU
                # print info and evaluation of selected batch
                if idx % CONFIG["SHOW_LOSS"] == CONFIG["SHOW_LOSS"] - 1:
                    avgLoss = runLoss / CONFIG["SHOW_LOSS"]
                    print("Epoch: %2d, iters: %4d/%d, loss: %.5f, runtime: %4.3f ms/iter, "
                          "Jaccard: %.2f, Precision: %.2f, Recall: %.2f"
                          % (epoch+1, idx+1, totalBatch, avgLoss, runtime, IU, precision, recall))
                    runLoss = 0.0
            # get evaluation
            logger["loss"].append(epochLoss / totalBatch)
            logger["precision"].append(runPrec / totalBatch)
            logger["recall"].append(runRec / totalBatch)
            logger["iou"].append(runIU / totalBatch)
            epochLoss, runPrec, runRec, runIU = 0.0, 0.0, 0.0, 0.0
            # save checkpoint
            if epoch not in [0, CONFIG["EPOCHS"] - 1] and epoch % CONFIG["SAVE_MODEL"] == 0:
                save_model(ckptDir, dcrf, epoch+1, logger, optimizer_DCRF)
        # save last checkpoint when finished training
        save_model(ckptDir, dcrf, epoch+1, logger, optimizer_DCRF)
        print("============================ DCRF Done Training ============================\n")

    ''' Validate DCRF '''

    if args.validate:
        # check DCRF checkpoint directory
        if str(args.checkpoint) != "checkpoint":
            if not args.checkpoint.is_file():
                raise IOError(f"Designated DCRF checkpoint file does not exist:\n"
                              f"{args.checkpoint.resolve()}")
            ckptPath = args.checkpoint.resolve()
        else:
            ckptDir = args.checkpoint.resolve()
            if not ckptDir.is_dir():
                raise IOError(f"Designated DCRF checkpoint folder does not exist:\n{ckptDir}")
            # get the lattest checkpoint if set to default directory
            fileList = sorted(ckptDir.glob("*.pt"),
                              reverse=True, key=lambda item: item.stat().st_ctime)
            if len(fileList) == 0:
                raise IOError(f"Cannot find any checkpoint files in:\n{ckptDir.resolve()}\n")

            ckptPath = fileList[0]
        # check DCRF image directory to read test samples
        if str(args.dir) != "dataset":
            if not args.dir.is_dir():
                raise IOError(f"Invalid sample folder to read from:\n{args.dir.resolve()}")

        baseDir = args.dir.joinpath("test")
        predDir = baseDir.joinpath("output")
        labDir = baseDir.joinpath("annotations")
        # output folder
        outDir = args.dir.parent.joinpath("results")
        outImgDir = outDir.joinpath("output")
        outRecDir = outDir.joinpath("evaluation")
        print(f"\nNow saving DCRF results to:\n{outDir.absolute()}")
        outDir.mkdir(exist_ok=True, parents=True)
        outImgDir.mkdir(exist_ok=True, parents=True)
        outRecDir.mkdir(exist_ok=True, parents=True)
        # DCRF test
        dcrf = dialated_crf(scale = CONFIG["SCALE"])
        checkpoint = torch.load(ckptPath)
        imgList = read_image_from_disk(predDir)
        assert len(imgList), "Empty folder"
        labList = read_image_from_disk(labDir, colorImg=False, isTensor=False)
        labList = trans_img_to_cls(labList)
        dcrf.load_state_dict(checkpoint['model_state_dict'])
        dcrf.eval()
        dcrf.to(device)
        cnt = 0
        totalTime = 0
        # get runtime estimation and save results to subfolder "results"
        with torch.no_grad():
            for name, img in imgList.items():
                cnt += 1
                img = img.unsqueeze(dim = 0)
                img = img.to(device)
                if cnt != 1: # skip the first image such that
                    start = time.time()
                    pred = dcrf(img)
                    end = time.time()
                    cTime = (end - start) * 1e3
                else:
                    pred = dcrf(img)
                    cTime = 0

                pred = pred[:, [1, 2, 0], :, :]
                pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
                pred = pred.detach().squeeze(0).cpu()
                lab = labList[name]
                save_image_to_disk(pred, outImgDir.joinpath(name))
                measure = Metrics(pred, lab, one_hot = False)
                measure.save_to_disk(name, outRecDir)
                totalTime += cTime
                print("Image: %4d/%d, Runtime: %6fms" %(cnt, len(imgList), cTime))

        print("\n============================ DCRF Validation Done ============================")
        print(f"Average inference time: {totalTime / (len(imgList) - 1 + 1e-31): .2f}ms")