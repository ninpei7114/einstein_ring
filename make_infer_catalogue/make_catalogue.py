import sys
import glob

import argparse
import astropy.io.fits
import numpy as np
import pandas as pd
import torch

sys.path.append("../")
from utils.ssd_model import nm_suppression


def calc_bbox(args, region, conf_thre):
    predict_bbox, scores = [], []
    detections = np.load(f"{args.result_path}/{region}/result.npy")
    position = np.load(f"{args.result_path}/{region}/position.npy")

    for d, p in zip(detections, position):
        conf_mask = d[1, :, 0] >= conf_thre
        detection_mask = d[1, :][conf_mask]
        if np.sum(conf_mask) >= 1:
            bbox = detection_mask[:, 1:] * np.array(int(p[2]))
            bbox = bbox + np.array([int(p[1]), int(p[0]), int(p[1]), int(p[0])])
            predict_bbox.append(bbox)
            scores.append(detection_mask[:, 0])

    bbox = torch.Tensor(np.concatenate(predict_bbox))
    scores = torch.Tensor(np.concatenate(scores))
    keep, count = nm_suppression(bbox, scores, overlap=0.3, top_k=5000)
    keep = keep[:count]
    bbox = bbox[keep]

    return bbox


def make_infer_catalogue(bbox, w):
    catalogue = pd.DataFrame(columns=["dec_min", "ra_min", "dec_max", "ra_max", "width_pix", "height_pix"])
    for i in bbox:
        width = int(i[2]) - int(i[0])
        height = int(i[3]) - int(i[1])
        GLONmax, GLATmin = w.all_pix2world(i[0], i[1], 0)
        GLONmin, GLATmax = w.all_pix2world(i[2], i[3], 0)
        temp = pd.DataFrame(
            columns=["dec_min", "ra_min", "dec_max", "ra_max", "width_pix", "height_pix"],
            data=[[GLATmin, GLONmin, GLATmax, GLONmax, width, height]],
            dtype="float64",
        )
        catalogue = pd.concat([catalogue, temp])

    return catalogue


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("result_path", type=str, help="model's path to infer")
    parser.add_argument("data_dir", help="LMC data path")

    return parser.parse_args()


def main(args):

    conf_thre = 0.5
    region_l = glob.glob(args.data_dir + "/fits_file/*")
    for region in region_l:
        hdu = astropy.io.fits.open(f"{args.data_dir}/{region.split('-')[0][3:]}/fits_file/{region}.fits")
        w = astropy.wcs.WCS(hdu.header)
        bbox = calc_bbox(args, region, conf_thre)
        catalogue = make_infer_catalogue(bbox, w)
        catalogue.to_csv(f"{args.result_path}/{region}/infer_catalogue.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
