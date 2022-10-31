import io
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

from global_configs.general_configs import (
    VAL_BATCH_SIZE,
    TEST_BATCH_SIZE
)

from utils import augs

from utils.aux_funcs import (
    get_train_val_split, calc_jaccard
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_tuples, batch_size, image_mask_augs, loader_type: str, logger: logging = None):
        self.image_mask_tuples = data_tuples  # [:100]
        self.batch_size = batch_size
        self.image_mask_augs = image_mask_augs()
        self.mask_augs = augs.mask_augs()
        self.loader_type = loader_type
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
        imgs, msks = btch_data[:, 0, ...], btch_data[:, 1, ...]
        for img, msk in zip(imgs, msks):
            aug_res = self.image_mask_augs(image=img.astype(np.uint8), mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # j_tries = 0
            # while True:
            msks_aug_res = self.mask_augs(image=img_aug.astype(np.uint8), mask=msk_aug.astype(np.uint8))
            _, msk_aug = msks_aug_res.get('image'), msks_aug_res.get('mask')

            j = calc_jaccard(msk, msk_aug)

            # if MIN_J < j < MAX_J or j_tries > MAX_J_TRIES:
            #     break
            # j_tries += 1

            btch_imgs_aug.append(tf.convert_to_tensor(img_aug, dtype=tf.float32))
            btch_msks_aug.append(tf.convert_to_tensor(msk_aug, dtype=tf.float32))
            btch_js.append(tf.convert_to_tensor(j, dtype=tf.float32))

        btch_imgs = np.array(btch_imgs_aug)
        btch_msks_aug = np.array(btch_msks_aug)
        btch_js = np.array(btch_js)

        # print(f'\nBatch which starts at indices [{start_index}:{end_index}) acquisition took: {get_runtime(seconds=time.time() - t_strt)} with {j_tries} retries')

        return (tf.convert_to_tensor(btch_imgs_aug, dtype=tf.float32), tf.convert_to_tensor(btch_msks_aug, dtype=tf.float32)), tf.convert_to_tensor(btch_js, dtype=tf.float32)


def get_data_loaders(data_tuples: list or np.ndarray, train_batch_size: int, train_augs, val_augs, test_augs, inf_augs, val_prop: float = .2, logger: logging.Logger = None):
    train_dl = val_dl = test_dl = inf_dl = None
    if train_batch_size > 0 and train_augs is not None and val_augs is not None:
        train_data, val_data = get_train_val_split(data_list=data_tuples, val_prop=val_prop, logger=logger)

        # - Create the DataLoader object
        train_dl = DataLoader(
            data_tuples=train_data,
            batch_size=train_batch_size,
            image_mask_augs=train_augs,
            loader_type='train',
            logger=logger
        )

        if isinstance(val_data, np.ndarray) and val_data.shape[0] > 0:
            val_dl = DataLoader(
                data_tuples=val_data,
                batch_size=VAL_BATCH_SIZE,
                image_mask_augs=train_augs,
                loader_type='validation',
                logger=logger
            )

    elif test_augs is not None:
        test_dl = DataLoader(
            data_tuples=data_tuples,
            batch_size=TEST_BATCH_SIZE,
            image_mask_augs=test_augs,
            loader_type='test',
            logger=logger
        )

    return train_dl, val_dl, test_dl


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
