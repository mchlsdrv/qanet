import io
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

from global_configs.general_configs import (
    VAL_BATCH_SIZE,
)

from utils import augs

from utils.aux_funcs import (
    get_train_val_split,
    calc_seg_measure, get_file_name, get_parent_dir_name, str_2_float, load_image, assert_pathable, str_2_path, check_pathable, get_files_under_dir,
)


def get_random_mask(masks_root: pathlib.Path or str, image_file: pathlib.Path or str):
    # - Assert the mask directory and the image file represent a path
    assert_pathable(argument=masks_root, argument_name='masks_root')
    assert_pathable(argument=image_file, argument_name='image_file')

    # - Make sure the mask directory and the image file are in the pathlib.Path format
    masks_root = str_2_path(path=masks_root)
    image_file = str_2_path(path=image_file)

    # - Get file name
    fl_name = get_file_name(path=image_file)

    # - Load the gt mask
    fl_msk_dir = masks_root / f'{get_parent_dir_name(path=image_file)}_{fl_name}'

    # - Get the masks which correspond to the current file
    msk_names = np.array(os.listdir(fl_msk_dir), dtype=object)
    n_msks = len(msk_names)

    # - Choose randomly n_samples masks
    rnd_idx = np.random.randint(0, n_msks)
    rnd_msk_name = msk_names[rnd_idx]

    # - Get the name of the mask
    rnd_msk_fl = fl_msk_dir / rnd_msk_name

    # - Get the seg measure of the chosen mask, which should be the name of the file with '_' instead of '.'
    j_str = get_file_name(path=rnd_msk_fl)

    # - Strip the '-' character in case it was added due to overlap in generated J values
    j = str_2_float(str_val=j_str if '-' not in j_str else j_str[:j_str.index('-')])

    # - Load the mask
    msk = load_image(image_file=str(rnd_msk_fl), add_channels=True)

    # - Return the random mask and the corresponding seg measure
    return msk, j


class DataLoader(tf.keras.utils.Sequence):
    """
    This object operates in two modes:

    1) Regular Mode:
    The image and its mask are used in an on-the-fly manner, where the mask is augmented to simulate a new segmentation, and
    the seg measure is calculated for each new sample batch. This mode is relatively slow.
    (*) To use this mode provide masks_dir=None.

    2) Preprocessed Mode:
    The masks should be produced and saved in advance in the following format -
        - a - Root dir containing the generated samples
        - b - All the generated masks for each image file will be placed in a new dir under the root, where the sub dirs of the file will be separated by '_'
        - c - Generated masks will have the seg measure as their name under the directory of the file
    This mode is much faster, as it doesn't require mask augmentation and the calculation of the seg measure.
    (*) To use this mode provide the images_root and masks_root arguments are valid pathable objects (i.e., pathlib.Path or str) containing the path to the
    root_dir in the aforementioned format.
    """
    def __init__(self, data_tuples: np.ndarray or list, file_tuples: np.ndarray or list, batch_size,  masks_dir: pathlib.Path or str = None, logger: logging = None):
        self.image_mask_tuples = data_tuples

        # - Ensure the batch_size is positive
        self.batch_size = batch_size if batch_size > 0 else 1

        self.image_mask_augs = augs.image_mask_augs()
        self.mask_augs = augs.mask_augs()
        self.transforms = augs.transforms()

        # - In case the masks are made in advance, in which case there is no need for the self.mask_augs
        self.image_files = np.array([fl_tpl[0] for fl_tpl in file_tuples], dtype=object)
        self.masks_dir = str_2_path(path=masks_dir)
        self.n_masks = 0 if not check_pathable(path=self.masks_dir) else len(get_files_under_dir(dir_path=self.masks_dir))

        self.n_smpls = self.n_masks if check_pathable(path=self.masks_dir) else len(self.image_mask_tuples)

        self.logger = logger

    def __len__(self):
        """
        > Returns the number of batches
        """
        if isinstance(self.masks_dir, pathlib.Path):
            # - Length in case of the masks made in advance is determined by their quantity
            data_length = int(np.floor(self.n_masks / self.batch_size))
        else:
            data_length = int(np.floor(len(self.image_mask_tuples) / self.batch_size))

        return data_length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size if start_idx + self.batch_size < self.n_smpls else self.n_smpls - 1

        if isinstance(self.masks_dir, pathlib.Path) and self.masks_dir.is_dir():
            item = self.preprocess_mode(start_index=start_idx, end_index=end_idx)
        # else:
        #     item = self.get_batch(start_index=start_idx, end_index=end_idx)

        return item

    def preprocess_mode(self, start_index, end_index):
        t_strt = time.time()

        btch_imgs_aug = []
        btch_msks_aug = []
        btch_js = []

        for img_fl in self.image_files[start_index:end_index]:
            # <1> Load the image
            img = load_image(image_file=img_fl, add_channels=True)

            # <2> Get the random augmentation and the corresponding seg measure
            msk, j = get_random_mask(masks_root=self.masks_dir, image_file=img_fl)

            # <3> Perform the image transformations
            aug_res = self.transforms(image=img.astype(np.uint8), mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # <4> Perform image and mask augmentations
            aug_res = self.image_mask_augs(image=img.astype(np.uint8), mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # - Add the data to the corresponding lists
            btch_imgs_aug.append(img_aug)
            btch_msks_aug.append(msk_aug)
            btch_js.append(j)

        # - Convert to tensors

        # - Images
        btch_imgs_aug = tf.convert_to_tensor(np.array(btch_imgs_aug), dtype=tf.float32)

        # - Masks
        btch_msks_aug = tf.convert_to_tensor(np.array(btch_msks_aug), dtype=tf.float32)

        # - Seg measures
        btch_js = tf.convert_to_tensor(btch_js, dtype=tf.float32)
        print(f'''
        btch_imgs_aug.shape = {btch_imgs_aug.shape}
        btch_msk_aug.shape = {btch_msks_aug.shape}
        btch_js.shape = {btch_js.shape}
        ''')
        return (btch_imgs_aug, btch_msks_aug), btch_js

    def get_batch(self, start_index, end_index):
        t_strt = time.time()
        btch_imgs_aug = []
        btch_msks_aug = []
        btch_js = []

        btch_data = self.image_mask_tuples[start_index:end_index, ...]
        btch_imgs, btch_msks = btch_data[:, 0, ...], btch_data[:, 1, ...]
        btch_msks_aug = []
        btch_msks_dfrmd = []
        for img, msk in zip(btch_imgs, btch_msks):
            # <1> Perform the image transformations
            aug_res = self.transforms(image=img.astype(np.uint8), mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # <2> Perform image and mask augmentations
            aug_res = self.image_mask_augs(image=img.astype(np.uint8), mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # <3> Perform mask augmentations
            msks_aug_res = self.mask_augs(image=img_aug, mask=msk_aug)
            _, msk_dfrmd = msks_aug_res.get('image'), msks_aug_res.get('mask')

            # - Add the data to the corresponding lists
            btch_imgs_aug.append(img_aug)
            btch_msks_aug.append(msk_aug)
            btch_msks_dfrmd.append(msk_dfrmd)

        # - Calculate the seg measure for the batch
        # <1> Convert the btch_msks_aug to numpy to calculate the seg measure
        btch_msks_aug = np.array(btch_msks_aug)
        # <2> Convert the btch_msks_dfrmd to numpy to calculate the seg measure
        btch_msks_dfrmd = np.array(btch_msks_dfrmd)
        # <3> Calculate the seg measure of the aug masks with the GT masks, and convert it to tensor
        btch_js = calc_seg_measure(gt_masks=btch_msks_aug, pred_masks=btch_msks_dfrmd)
        # <4> Convert btch_js to tensor
        btch_js = tf.convert_to_tensor(btch_js, dtype=tf.float32)

        # - Convert the btch_imgs_aug to numpy array and then to tensor
        btch_imgs_aug = tf.convert_to_tensor(np.array(btch_imgs_aug), dtype=tf.float32)

        # - Convert the btch_masks_aug to tensor right away, as it was already converted to numpy in <1>
        btch_msks_dfrmd = tf.convert_to_tensor(btch_msks_dfrmd, dtype=tf.float32)

        return (btch_imgs_aug, btch_msks_dfrmd), btch_js


def get_data_loaders(data_tuples: list or np.ndarray, file_tuples: list or np.ndarray, masks_dir, train_batch_size: int, val_prop: float = .2, logger: logging.Logger = None):
    train_dl = val_dl = test_dl = inf_dl = None
    train_data, val_data = get_train_val_split(data_list=data_tuples, val_prop=val_prop, logger=logger)
    train_files, val_files = get_train_val_split(data_list=file_tuples, val_prop=val_prop, logger=logger)

    # - Create the DataLoader object
    train_dl = DataLoader(
        data_tuples=train_data,
        file_tuples=train_files,
        batch_size=train_batch_size,
        masks_dir=masks_dir,
        logger=logger
    )

    if isinstance(val_data, np.ndarray) and val_data.shape[0] > 0:
        val_dl = DataLoader(
            data_tuples=val_data,
            file_tuples=val_files,
            batch_size=VAL_BATCH_SIZE,
            masks_dir=masks_dir,
            logger=logger
        )

    return train_dl, val_dl


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
