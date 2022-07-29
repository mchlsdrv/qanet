import os
import time

import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pathlib
import logging

from configs.general_configs import SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX, VAL_BATCH_SIZE, TEST_BATCH_SIZE, NUM_WORKERS, GEN_DATA_FILE, N_SAMPLES
from custom import augs
from utils import aux_funcs  # import calc_jaccard, get_runtime

from utils.logging_funcs import (
    info_log,
)

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


class ImageDS(Dataset):
    def __init__(self, data_tuples, augs):
        self.data_tuples = data_tuples
        self.augs = augs

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, index):
        img, mask, jaccard = self.data_tuples[index]
        aug_res = self.augs(image=img, mask=mask)
        img, mask = aug_res.get('image'), aug_res.get('mask').unsqueeze(0)

        return img, mask, torch.tensor(jaccard, dtype=torch.float)
    # def __init__(self, file_tuples, augs):
    #     self.file_tuples = file_tuples
    #     self.augs = augs
    #     self.mask_augs = mask_augs()
    #
    # def __len__(self):
    #     return len(self.file_tuples)
    #
    # def __getitem__(self, index):
    #     img = np.array(Image.open(str(self.file_tuples[index][0])).convert('L'), dtype=np.uint8)
    #     mask = np.array(Image.open(str(self.file_tuples[index][1])).convert('L'), dtype=np.float32)
    #
    #     aug_res = self.augs(image=img, mask=mask)
    #     img, mask = aug_res.get('image'), aug_res.get('mask').unsqueeze(0)
    #     # print(f'img_aug.shape: {img_aug.shape}, mask_aug.shape: {mask_aug.shape}')
    #
    #     mask_aug_res = self.mask_augs(image=np.array(img), mask=np.array(mask))
    #     mask_aug = mask_aug_res.get('mask')
    #
    #     jaccard = aux_funcs.calc_jaccard(R=mask, S=mask_aug)
    #
    #     return img, mask, mask_aug, torch.tensor(jaccard, dtype=torch.float)
    

def get_train_val_split(data_list: list or np.ndarray, val_prop: float = .2, logger: logging.Logger = None):
    n_items = len(data_list)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * val_prop)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data_list, dtype=object)

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    # - The items for training are the once which are not included in the validation set
    train_data = np_data[np.setdiff1d(item_idxs, val_idxs)]

    info_log(logger=logger, message=f'| Number of train data files : {len(train_data)} | Number of validation data files : {len(val_data)} |')

    return train_data, val_data


def scan_files(root_dir: pathlib.Path or str, seg_dir_postfix: str, image_prefix: str, seg_prefix: str):
    file_tuples = list()
    for root, dirs, _ in os.walk(root_dir):
        root = pathlib.Path(root)
        for dir in dirs:
            seg_dir = root / f'{dir}_{seg_dir_postfix}'
            if seg_dir.is_dir():
                for sub_root, _, files in os.walk(root / dir):
                    for file in files:
                        img_fl = pathlib.Path(f'{sub_root}/{file}')
                        seg_fl = pathlib.Path(f'{seg_dir}/{file.replace(image_prefix, seg_prefix)}')
                        if img_fl.is_file() and seg_fl.is_file():
                            file_tuples.append((img_fl, seg_fl))
    return file_tuples


def generate_data(files: list, save_dir: pathlib.Path, n_samples=N_SAMPLES, seg_dir_postfix=SEG_DIR_POSTFIX, image_prefix=IMAGE_PREFIX, seg_prefix=SEG_PREFIX, plot_samples: bool = False):
    t_start = time.time()
    mask_augs = augs.mask_augs()
    imgs = []
    masks = []
    aug_masks = []
    jaccards = []

    n_passes = n_samples // len(files) if len(files) > 0 else 0
    for pass_idx in range(n_passes):
        print(f'== Pass: {pass_idx+1}/{n_passes} ==')
        files_pbar = tqdm(files)
        for img_fl, seg_fl in files_pbar:
            img = cv2.imread(str(img_fl), -1)
            mask = cv2.imread(str(seg_fl), -1)

            aug_res = mask_augs(image=img, mask=mask)
            img_aug, mask_aug = aug_res.get('image'), aug_res.get('mask')

            jaccard = aux_funcs.calc_jaccard(mask, mask_aug)

            if 0 < jaccard < 1:
                imgs.append(img)
                masks.append(mask)
                aug_masks.append(mask_aug)
                jaccards.append(jaccard)
                files_pbar.set_postfix(jaccard=f'{jaccard:.4f}')

    data = np.array(list(zip(imgs, masks, aug_masks, jaccards)), dtype=object)

    # - Save the data

    np.save(str(save_dir / f'data_{n_samples}_samples.npy'), data, allow_pickle=True)

    # - Plot samples
    samples_dir = save_dir / 'samples'
    os.makedirs(samples_dir, exist_ok=True)

    if plot_samples:
        for idx, (img, msk, msk_aug, j) in enumerate(data):
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(img, cmap='gray')
            ax[1].imshow(msk, cmap='gray')
            ax[2].imshow(msk_aug, cmap='gray')
            fig.suptitle(f'Jaccard: {j:.3f}')

            plt.savefig(samples_dir / f'{idx}.png')
            plt.close(fig)

    # - Plot J histogram
    heights, ranges = np.histogram(jaccards, range=[0., 1.], bins=10, density=True)
    fig, ax = plt.subplots()
    ax.bar(ranges[:-1], heights)
    ax.set(xticks=np.arange(0.0, 1.1, 0.1), xlim=[0, 1])
    plt.savefig(save_dir / 'jaccard dist.png')

    plt.close(fig)

    print(f'Data generation took: {aux_funcs.get_runtime(seconds=time.time() - t_start)}')

    return data


def get_data_loaders(data_dir: str or pathlib.Path, train_batch_size: int, train_augs, val_augs, test_augs, seg_dir_postfix: str = SEG_DIR_POSTFIX, image_prefix: str = IMAGE_PREFIX, seg_prefix: str = SEG_PREFIX, val_prop: float = .2, logger: logging.Logger = None):
    fls = scan_files(
        root_dir=data_dir,
        seg_dir_postfix=seg_dir_postfix,
        image_prefix=image_prefix,
        seg_prefix=seg_prefix
    )

    if GEN_DATA_FILE.is_file():
        data = np.load(str(GEN_DATA_FILE), allow_pickle=True)
    else:
        data = generate_data(files=fls, save_dir=GEN_DATA_FILE.parent)

    train_dl = val_dl = test_dl = None
    if train_augs is not None and val_augs is not None:
        train_data, val_data = get_train_val_split(data_list=data, val_prop=val_prop, logger=logger)

        # - Create the DataLoader object
        train_dl = DataLoader(
            ImageDS(data_tuples=train_data, augs=train_augs()),
            batch_size=train_batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

        if val_data is not None and len(val_data):
            val_dl = DataLoader(
                ImageDS(data_tuples=val_data, augs=val_augs()),
                batch_size=VAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                shuffle=True
            )
    elif test_augs is not None:
        test_dl = DataLoader(
            ImageDS(data_tuples=data, augs=test_augs()),
            batch_size=TEST_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

    return train_dl, val_dl, test_dl
