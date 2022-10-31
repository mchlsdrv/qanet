import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from global_configs.general_configs import (
    VAL_PROP,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    NUM_TRAIN_WORKERS,
    NUM_VAL_WORKERS,
    PIN_MEMORY,
)
from utils.aux_funcs import get_train_val_split

__author__ = 'sidorov@post.bgu.ac.il'


class ImageDS(Dataset):
    def __init__(self, data_tuples, augs):
        self.data_tuples = data_tuples
        self.augs = augs

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        img, gt_mask, aug_mask, jaccard = self.data_tuples[index]
        aug_res = self.augs(image=img.astype(np.uint8), mask=aug_mask)
        img, mask = aug_res.get('image'), aug_res.get('mask')
        img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)

        return torch.tensor(img, dtype=torch.float), torch.tensor(mask.astype(np.int16), dtype=torch.float), torch.tensor(jaccard, dtype=torch.float)


def get_data_loaders(data_file: str or pathlib.Path, train_augs, val_augs, logger: logging.Logger = None):
    # - Load the data
    data = np.load(str(data_file), allow_pickle=True)

    # - Split data into train / validation datasets
    train_data, val_data = get_train_val_split(data_list=data, val_prop=VAL_PROP, logger=logger)

    # - Create the train / validation dataloaders
    train_dl = DataLoader(
        ImageDS(data_tuples=train_data, augs=train_augs()),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_TRAIN_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    val_dl = DataLoader(
        ImageDS(data_tuples=val_data, augs=val_augs()),
        batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_VAL_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )

    return train_dl, val_dl
