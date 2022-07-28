import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import pickle as pkl
import numpy as np
import pathlib
import logging

from configs.general_configs import SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX, VAL_BATCH_SIZE, TEST_BATCH_SIZE, NUM_WORKERS
from custom.augs import (
    mask_augs,
)

from utils.logging_funcs import (
    info_log,
)

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


class ImageDS(Dataset):
    def __init__(self, file_tuples, augs):
        self.file_tuples = file_tuples
        self.augs = augs
        self.mask_augs = mask_augs()

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, index):
        img = np.array(Image.open(str(self.file_tuples[index][0])).convert('L'), dtype=np.uint8)
        mask = np.array(Image.open(str(self.file_tuples[index][1])).convert('L'), dtype=np.float32)

        aug_res = self.augs(image=img, mask=mask)
        img, mask = aug_res.get('image'), aug_res.get('mask').unsqueeze(0)
        # print(f'img_aug.shape: {img_aug.shape}, mask_aug.shape: {mask_aug.shape}')

        mask_aug_res = self.mask_augs(image=np.array(img), mask=np.array(mask))
        mask_aug = mask_aug_res.get('mask')

        jaccard = calc_jaccard(R=mask, S=mask_aug)

        return img, mask, mask_aug, torch.tensor(jaccard, dtype=torch.float)


def calc_jaccard(R: np.ndarray, S: np.ndarray):
    """
    Calculates the mean Jaccard coefficient for two multi-class labels
    :param: R - Reference multi-class label
    :param: S - Segmentation multi-class label
    """
    def _get_one_hot_masks(multi_class_mask: np.ndarray, classes: np.ndarray = None):
        """
        Converts a multi-class label into a one-hot labels for each object in the multi-class label
        :param: multi_class_mask - mask where integers represent different objects
        """
        # - Ensure the multi-class label is populated with int values
        mlt_cls_mask = multi_class_mask.astype(np.int16)

        # - Find the classes
        cls = classes
        if cls is None:
            cls = np.unique(mlt_cls_mask)

        # - Discard the background (0)
        cls = cls[cls > 0]

        one_hot_masks = np.zeros((len(cls), *mlt_cls_mask.shape), dtype=np.float32)
        for lbl_idx, lbl in enumerate(one_hot_masks):
            # for obj_mask_idx, _ in enumerate(lbl):
            idxs = np.argwhere(mlt_cls_mask == cls[lbl_idx])
            x, y = idxs[:, 0], idxs[:, 1]
            one_hot_masks[lbl_idx, x, y] = 1.
        return one_hot_masks

    # - In case theres no other classes besides background (i.e., 0) J = 0.
    J = 0.
    R = np.array(R)
    S = np.array(S)
    # - Convert the multi-label mask to multiple one-hot masks
    cls = np.unique(R.astype(np.int16))

    if cls[cls > 0].any():  # <= If theres any classes besides background (i.e., 0)
        R = _get_one_hot_masks(multi_class_mask=R, classes=cls)
        S = _get_one_hot_masks(multi_class_mask=S, classes=cls)

        # - Calculate the intersection of R and S
        I_sums = np.sum(R[:, np.newaxis, ...] * S[np.newaxis, ...], axis=(-2, -1))
        x, y = np.arange(len(I_sums)), np.argmax(I_sums, axis=1)  # <= Choose the once that have the largest overlap with the ground truth label
        I = I_sums[x, y]

        # - Calculate the union of R and S
        U_sums = np.sum(np.logical_or(R[:, np.newaxis, ...], S[np.newaxis, ...]), axis=(-2, -1))
        U = U_sums[x, y]

        # - Mean Jaccard on the valid items only
        U[U <= 0] = 1  # <= To avoid division by 0
        J = (I / U)

        # - Calculate the areas of the reference items
        R_areas = R.sum(axis=(-2, -1))
        R_areas[R_areas <= 0] = 1  # <= To avoid division by 0

        # - Find out the indices of the items which do not satisfy |I| / |R| > 0.5 and replace them with 0
        inval = np.argwhere((I / R_areas) <= .5).reshape(-1)

        J[inval] = np.nan

        J = np.nan_to_num(J, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    return J if isinstance(J, float) else J.mean()


def get_files_from_metadata(root_dir: str or pathlib.Path, metadata_files_regex, logger: logging.Logger = None):
    img_seg_fls = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if re.match(metadata_files_regex, file) is not None:
                metadata_file = f'{root}/{file}'
                with pathlib.Path(metadata_file).open(mode='rb') as pkl_in:
                    metadata = pkl.load(pkl_in)
                    for metadata_tuple in metadata.get('filelist'):
                        img_seg_fls.append((f'{root}/{metadata_tuple[0]}', f'{root}/{metadata_tuple[1]}'))

    return img_seg_fls


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


def get_data_loaders(data_dir: str or pathlib.Path, train_batch_size: int, train_augs, val_augs, test_augs, seg_dir_postfix: str = SEG_DIR_POSTFIX, image_prefix: str = IMAGE_PREFIX, seg_prefix: str = SEG_PREFIX, val_prop: float = .2, logger: logging.Logger = None):
    fls = scan_files(
        root_dir=data_dir,
        seg_dir_postfix=seg_dir_postfix,
        image_prefix=image_prefix,
        seg_prefix=seg_prefix
    )

    train_dl = val_dl = test_dl = None
    if train_augs is not None and val_augs is not None:
        train_fls, val_fls = get_train_val_split(data_list=fls, val_prop=val_prop, logger=logger)

        # - Create the DataLoader object
        train_dl = DataLoader(
            ImageDS(file_tuples=train_fls, augs=train_augs()),
            batch_size=train_batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

        if val_fls is not None and len(val_fls):
            val_dl = DataLoader(
                ImageDS(file_tuples=val_fls, augs=val_augs()),
                batch_size=VAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                shuffle=True
            )
    elif test_augs is not None:
        test_dl = DataLoader(
            ImageDS(file_tuples=fls, augs=test_augs()),
            batch_size=TEST_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

    return train_dl, val_dl, test_dl
