import glob
import cv2

import numpy as np
import torch
from torch.nn import functional as F


def resize(data, size):
    """
    sizeは、自由
    今はy ,xは同じサイズだが、違うサイズにしたければ、タプルでsizeを入力するとよい
    入力データ:（y, x, 2 or 3）
    出力:（size ,size, 2 or 3）
    """
    cut_data = np.swapaxes(data, 1, 2)
    cut_data = np.swapaxes(cut_data, 0, 1)
    cut_data = torch.from_numpy(cut_data)
    cut_data = cut_data.unsqueeze(0)
    resize_data = F.interpolate(cut_data, (size, size), mode="bilinear", align_corners=False)
    resize_data = np.squeeze(resize_data.detach().numpy())

    resize_data_ = np.swapaxes(resize_data, 0, 1)
    resize_data_ = np.swapaxes(resize_data_, 1, 2)
    return resize_data_


def resize_png(dirname):
    """
    dirname: str
    """

    png_files = glob.glob(dirname + "/*.png")
    for png_file in png_files:
        data = cv2.imread(png_file)
        resize_data_ = resize(data, 300)
        cv2.imwrite(png_file, resize_data_)
