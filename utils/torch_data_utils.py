import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import pickle as pkl
import numpy as np
import pathlib
import logging

from configs.torch_general_configs import SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX
from custom.torch_augs import (
    mask_augs,
)

# from utils.torch_image_funcs import (
#     get_contours,
# )

from utils.torch_logging_funcs import (
    info_log,
)

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


class ImageDS(Dataset):
    def __init__(self, file_tuples, augs):
        self.file_tuples = file_tuples
        self.augs = augs
        self.mask_augs = mask_augs()

    # def _clean_blanks(self):
    #     """
    #     Cleans all the images there theres no data, i.e., blank or only random noise
    #     """
    #     clean_data = []
    #     for idx, (img, seg) in enumerate(self.imgs_segs):
    #         bin_seg = copy(seg)
    #
    #         # - Find the indices of the current labels' class in the segmentation label
    #         seg_px = np.arguer(bin_seg > 0)
    #
    #         # - Separate the indices into x and y coordinates
    #         seg_pix_xs, seg_pix_ys = seg_px[:, 0], seg_px[:, 1]
    #
    #         # - Mark the entries at the indices that correspond to the label as '1', to produce a binary label
    #         bin_seg[(seg_pix_xs, seg_pix_ys)] = 1
    #
    #         # - Images should be of type UINT8
    #         bin_seg = bin_seg.astype(np.uint8)
    #
    #         # - Find the contours in the image
    #         _, centroids = get_contours(image=bin_seg)
    #
    #         # - If the image has at least one contour (i.e., it is not blank or noise) - add it to the data
    #         if centroids:
    #             clean_data.append((img, seg))
    #
    #     self.imgs_segs = np.array(clean_data)

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

        jaccard = get_jaccard(mask, mask_aug)

        return img, mask, mask_aug, jaccard


def get_jaccard(gt, pred):

    gt[gt > 0] = 1
    pred[pred > 0] = 1

    I = np.logical_and(gt, pred)
    U = np.logical_and(gt, pred)
    J = I.sum() / (U.sum() + 1e-8)

    return J


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


def get_data_loaders(data_dir: str or pathlib.Path, batch_size: int, train_augs, val_augs, test_augs, seg_dir_postfix: str = SEG_DIR_POSTFIX, image_prefix: str = IMAGE_PREFIX, seg_prefix: str = SEG_PREFIX, val_prop: float = .2, logger: logging.Logger = None):
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
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True
        )

        if val_fls is not None and len(val_fls):
            val_dl = DataLoader(
                ImageDS(file_tuples=val_fls, augs=val_augs()),
                batch_size=1,
                num_workers=2,
                pin_memory=True,
                shuffle=True
            )
    elif test_augs is not None:
        test_dl = DataLoader(
            ImageDS(file_tuples=fls, augs=test_augs()),
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=True
        )

    return train_dl, val_dl, test_dl
