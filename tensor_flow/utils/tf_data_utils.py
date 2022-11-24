import io
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
    calc_seg_measure,
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_tuples, batch_size, logger: logging = None):
        self.image_mask_tuples = data_tuples  # [:100]
        self.batch_size = batch_size
        self.image_mask_augs = augs.image_mask_augs()
        self.mask_augs = augs.mask_augs()
        self.transforms = augs.transforms()
        self.logger = logger

    def __len__(self):
        """
        > Returns the number of batches
        """
        return int(np.floor(len(self.image_mask_tuples) / self.batch_size)) if self.batch_size > 0 else 0

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size if start_idx + self.batch_size < len(self.image_mask_tuples) else len(self.image_mask_tuples) - 1
        return self.get_batch(start_index=start_idx, end_index=end_idx)

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


def get_data_loaders(data_tuples: list or np.ndarray, train_batch_size: int, val_prop: float = .2, logger: logging.Logger = None):
    train_dl = val_dl = test_dl = inf_dl = None
    train_data, val_data = get_train_val_split(data_list=data_tuples, val_prop=val_prop, logger=logger)

    # - Create the DataLoader object
    train_dl = DataLoader(
        data_tuples=train_data,
        batch_size=train_batch_size,
        logger=logger
    )

    if isinstance(val_data, np.ndarray) and val_data.shape[0] > 0:
        val_dl = DataLoader(
            data_tuples=val_data,
            batch_size=VAL_BATCH_SIZE,
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
