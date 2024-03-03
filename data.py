import tarfile
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import webdataset


## webdatasetのために作成
def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = np.array(imgs)

    return imgs, targets


## webdatasetのために作成
def preprocess(sample):
    img, json = sample
    return np.array(img) / 255, [
        (float(x["XMin"]), float(x["YMin"]), float(x["XMax"]), float(x["YMax"]), float(x["Confidence"])) for x in json
    ]


# 無限イテレータ
def InfiniteIterator(loader):
    iter = loader.__iter__()
    while True:
        try:
            x = next(iter)
        except StopIteration:
            iter = loader.__iter__()  # 終わっていたら最初に戻る
            x = next(iter)
        yield x


class DataSet:
    def __init__(self, data, label):
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class NegativeSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, source, true_size, sample_negative_size):
        self.true_size = true_size
        self.negative_size = len(source) - true_size
        self.sample_negative_size = sample_negative_size

    def __iter__(self):
        neg = np.arange(self.true_size, self.true_size + self.sample_negative_size)
        indeces = np.concatenate((np.arange(self.true_size), np.random.choice(neg, self.sample_negative_size)))
        np.random.shuffle(indeces)
        for i in indeces:
            yield i

    def __len__(self):
        return self.true_size + self.sample_negative_size


def make_training_dataloader(args, train_Ring_num, nonring_num):
    with tarfile.open(f"{args.train_val_data_path}/bubble_dataset_train_ring.tar", "w:gz") as tar:
        tar.add(f"{args.train_val_data_path}/train/ring")
    ## Training Ring の Dataloader を作成
    Training_Ring_web = (
        webdataset.WebDataset(f"{args.train_val_data_path}/bubble_dataset_train_ring.tar")
        .shuffle(10000000)
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
    )
    dl_ring_train = torch.utils.data.DataLoader(
        Training_Ring_web,
        collate_fn=od_collate_fn,
        batch_size=args.Ring_mini_batch,
        num_workers=2,
        pin_memory=True,
    )

    with tarfile.open(f"{args.train_val_data_path}/bubble_dataset_train_nonring.tar", "w:gz") as tar:
        tar.add(f"{args.train_val_data_path}/train/nonring/")
    NonRing_rsample = train_Ring_num / nonring_num
    NonRing_web_list = (
        webdataset.WebDataset(f"{args.train_val_data_path}/bubble_dataset_train_nonring.tar")
        .rsample(NonRing_rsample)
        .shuffle(10000000000)
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
    )

    NonRing_dl_l = torch.utils.data.DataLoader(
        NonRing_web_list, collate_fn=od_collate_fn, batch_size=args.NonRing_mini_batch, num_workers=2, pin_memory=True
    )

    return dl_ring_train, NonRing_dl_l  # NonRingを無限にループするイテレータへ


def make_validatoin_dataloader(args):
    with tarfile.open(f"{args.train_val_data_path}/bubble_dataset_val.tar", "w:gz") as tar:
        tar.add(f"{args.train_val_data_path}/val")
    Dataset_val = (
        webdataset.WebDataset(f"{args.train_val_data_path}/bubble_dataset_val.tar")
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
    )
    dl_val = torch.utils.data.DataLoader(
        Dataset_val,
        collate_fn=od_collate_fn,
        batch_size=args.Val_mini_batch,
        num_workers=2,
        pin_memory=True,
    )

    return dl_val
