import argparse
import itertools
import os
import shutil
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.optim as optim
import wandb
from PIL import ImageFile
from train_model import train_model
from training_sub import print_and_log, weights_init
from utils.ssd_model import SSD, MultiBoxLoss


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument(
        "train_val_data_path",
        metavar="DIR",
        help="validation data path",
    )

    parser.add_argument("--savedir_path", metavar="DIR", default="/workspace/weights/search/", help="savedire path")
    # minibatch
    parser.add_argument("--num_epoch", type=int, default=300, help="number of total epochs to run (default: 300)")
    parser.add_argument("--Ring_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--NonRing_mini_batch", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--Val_mini_batch", default=128, type=int, help="Validation mini-batch size (default: 128)")
    parser.add_argument("--True_iou", default=0.5, type=float, help="True IoU in MultiBoxLoss(default: 0.5)")
    # 学習率
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--init_random_state", type=int, default=123)
    # option
    parser.add_argument("--test_infer_false", action="store_false")
    parser.add_argument("--ring_select_false", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="リングの選定")
    parser.add_argument("--wandb_name", type=str, default="search_validation_size")

    return parser.parse_args()


# SSDの学習
def main(args):
    """SSDの学習を行う。

    :Example command:
    >>> python train_main.py /dataset/spitzer_data --savedir_path /workspace/webdataset_weights/Ring_selection_compare/ \
        --NonRing_data_path /workspace/NonRing_png/region_NonRing_png/ \
        --validation_data_path /workspace/cut_val_png/region_val_png/ \
        -s -i 0 --NonRing_remove_class_list 3 --Ring_mini_batch 16 --NonRing_mini_batch 2 --Val_mini_batch 64 \
        --l18_infer --ring_select

    """
    torch.manual_seed(args.init_random_state)
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if os.path.exists(args.savedir_path):
        print("REMOVE FILES...")
        shutil.rmtree(args.savedir_path)
    os.makedirs(args.savedir_path, exist_ok=True)

    ## 上下反転、回転、縮小、平行移動の4パターンの組み合わせでaugmentationをする。
    flip_list = [True]  # , False]
    rotate_list = [True]  # , False]
    scale_list = [False]
    translation_list = [True]

    for flip, rotate, scale, translation in itertools.product(flip_list, rotate_list, scale_list, translation_list):
        train_cfg = {"flip": flip, "rotate": rotate, "scale": scale, "translation": translation}
        name_ = []
        [name_.append(k + "_" + str(v) + "__") for k, v in zip(list(train_cfg.keys()), list(train_cfg.values()))]
        name = args.savedir_path + "/" + "".join(name_)
        os.makedirs(name, exist_ok=True)

        ############
        ## logger ##
        ############
        f_log = open(name + "/log.txt", "w")
        log_list = [
            "#######################",
            "augmentation parameter",
            "#######################",
            f"flip: {flip}, rotate: {rotate}, scale: {scale}, translation: {translation}",
            " ",
            "#######################",
            "   args parameters",
            "#######################",
            f"learning_rate: {args.lr}",
            f"weight_decay: {args.weight_decay}",
            f"Ring_mini_batch: {args.Ring_mini_batch}",
            f"NonRing_mini_batch: {args.NonRing_mini_batch,}",
            " ",
            "====================================",
        ]
        print_and_log(f_log, log_list)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "Ring_mini_batch": args.Ring_mini_batch,
                "NonRing_mini_batch": args.NonRing_mini_batch,
            },
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_and_log(f_log, f"使用デバイス： {device}")

        net = SSD()
        ## パラメータを初期化
        for net_sub in [net.vgg, net.extras, net.loc, net.conf]:
            net_sub.apply(weights_init)
        net.to(device)
        wandb.watch(net, log_freq=100)

        optimizer = optim.AdamW(
            net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False
        )
        train_model_params = {
            "net": net,
            "criterion": MultiBoxLoss(jaccard_thresh=args.True_iou, neg_pos=3, device=device),
            "optimizer": optimizer,
            "num_epochs": args.num_epoch,
            "f_log": f_log,
            "augmentation_name": name,
            "args": args,
            "train_cfg": train_cfg,
            "device": device,
            "run": run,
        }

        ####################
        ## Training Model ##
        ####################
        train_model(**train_model_params)

        f_log.close()
        artifact = wandb.Artifact("training_log", type="dir")
        artifact.add_dir(name)
        run.log_artifact(artifact, aliases=["latest", "best"])

        run.alert(title="学習が終了しました", text="学習が終了しました")
        run.finish()

        shutil.rmtree(args.savedir_path)
        os.makedirs(args.savedir_path, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
