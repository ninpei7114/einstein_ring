import glob
import os
import shutil
import time
from itertools import product as product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import make_training_dataloader, make_validatoin_dataloader
from make_figure import make_figure
from training_sub import (
    EarlyStopping_f1_score,
    EarlyStopping_loss,
    management_loss,
    print_and_log,
    write_train_log,
)
from utils.ssd_model import Detect


def train_model(
    net, criterion, optimizer, num_epochs, f_log, augmentation_name, args, train_cfg, device, run, val_size
):
    """モデルの学習を実行する関数

    Args:
        net (pytorch Modulelist): SSDネットワーク
        criterion (MultiBoxLoss): 損失関数
        optimizer (AdamW)       : 最適化手法
        num_epochs (int)        : 最大epoch数
        f_log (txt file)        : logファイル
        augmentation_name (int) : どのaugmentationを使用したかの名前
        args (args)             : argparseの引数
        train_cfg (dictionary)  : augmentationのパラメータ
        device (torch.device)   : GPU or CPU
    """
    early_stopping = EarlyStopping_loss(
        patience=15, verbose=True, path=augmentation_name + "/earlystopping.pth", flog=f_log
    )

    detect = Detect(nms_thresh=0.45, top_k=500, conf_thresh=0.5)  # F1 scoreのconfの計算が0.3からなので、ここも0.3
    save_training_val_loss = management_loss()
    logs = []

    ##########################################
    ## Training data & Validation dataの作成 ##
    ##########################################

    Val_num = len(glob.glob(f"{args.train_val_data_path}/val/*.png"))
    NonRing_num, train_Ring_num = len(glob.glob(f"{args.train_val_data_path}/train/non_ring/*.png")), len(
        glob.glob(f"{args.train_val_data_path}/train/ring/*.png")
    )
    dl_ring_train, dl_nonring = make_training_dataloader(args, train_Ring_num, NonRing_num)
    dl_val = make_validatoin_dataloader(args)
    all_iter_val = int(int(Val_num) / args.Val_mini_batch)

    for epoch in range(num_epochs):
        start_time = time.time()
        iteration_train, iteration_val = 0, 0
        save_training_val_loss()  # lossの初期化
        print_and_log(f_log, ["-------------", "Epoch {}/{}".format(epoch + 1, num_epochs), "-------------"])

        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}
        all_iter = int(int(train_Ring_num) / args.Ring_mini_batch)

        #############
        ## 学習開始 ##
        #############
        for phase in ["train", "val"]:
            if phase == "train":
                print_and_log(f_log, f" ({phase}) ")
                iter_noring = dl_nonring.__iter__()
                net.train()
            else:
                print_and_log(f_log, f" \n ({phase})")
                net.eval()
                result, position, regions = [], [], []

            ############################
            ## データの整形とモデルに入力 ##
            ############################
            for _ in dataloaders_dict[phase]:
                if phase == "train":
                    images, targets = _[0], _[1]
                    noring = next(iter_noring, None)
                    if noring is None:
                        iter_noring = dl_nonring.__iter__()
                        noring = next(iter_noring)
                    images = np.concatenate((images, noring[0]))
                    targets = targets + noring[1]
                else:
                    images, targets, offset, region_info = _[0], _[1], _[2], _[3]

                images = torch.from_numpy(images).permute(0, 3, 1, 2)[:, :2, :, :]
                images = images.to(device, dtype=torch.float)
                targets = [ann.to(device, dtype=torch.float) for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, decoded_box = net(images)
                    loss_dic = criterion(outputs, targets)
                    loss = loss_dic["loc_loss"] + loss_dic["conf_loss"]

                    if phase == "train":
                        loss.backward()  # 勾配の計算
                        # 勾配が大きくなりすぎると計算が不安定になるため、clipで最大でも勾配10.0に留める
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
                        optimizer.step()  # パラメータ更新
                        print("\r" + str(iteration_train) + "/" + str(all_iter) + " ", end="")
                        iteration_train += 1
                        save_training_val_loss.sum_iter_loss(loss_dic, "train")
                    else:
                        print("\r" + str(iteration_val) + "/" + str(all_iter_val) + " ", end="")
                        iteration_val += 1
                        result.append(detect(*outputs).to("cpu").detach().numpy().copy())
                        position.extend(offset)
                        regions.extend(region_info)
                        save_training_val_loss.sum_iter_loss(loss_dic, "val")

        ###############
        ## Lossの管理 ##
        ###############
        # loc, confのlossを出力
        loss_train = save_training_val_loss.output_each_loss("train", iteration_train)
        loss_val = save_training_val_loss.output_each_loss("val", iteration_val)

        log_epoch = write_train_log(f_log, epoch, loss_train, loss_val, start_time)
        run.log(log_epoch)
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(augmentation_name + "/log_output.csv")

        early_stopping(loss_val["loc_loss"] + loss_val["conf_loss"], net, epoch, optimizer, loss_train)
        # early_stopping(f_score_val, net, epoch, optimizer, loss_train, loss_val)
        if early_stopping.early_stop:
            print_and_log(f_log, "Early_Stopping")
            break

    ## lossの推移を描画する
    loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s = save_training_val_loss.output_all_epoch_loss()
    make_figure(augmentation_name, loc_l_val_s, conf_l_val_s, loc_l_train_s, conf_l_train_s)
