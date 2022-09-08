import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
from global_configs.general_configs import VAL_BATCH_SIZE, TEST_BATCH_SIZE

from utils.aux_funcs import (
    get_train_val_split,
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_tuples, batch_size, augs, loader_type: str, logger: logging = None):
        self.data_tuples = data_tuples
        self.batch_size = batch_size
        self.augs = augs()
        self.loader_type = loader_type
        self.logger = logger

    def __len__(self):
        """
        > Returns the number of batches
        """
        return int(np.floor(len(self.data_tuples) / self.batch_size)) if self.batch_size > 0 else 0

    def __getitem__(self, index):
        data_item = None

        if self.loader_type in ['train', 'validation', 'test']:
            img, _, aug_mask, jaccard = self.data_tuples[index]
            aug_res = self.augs(image=img.astype(np.uint8), mask=aug_mask)
            img, mask = aug_res.get('image'), aug_res.get('mask')
            img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)
            jaccard = np.array([jaccard])

            data_item = (tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32)), (tf.convert_to_tensor(jaccard, dtype=tf.float32))

        if self.loader_type == 'inference':
            img_fl, mask_fl = self.data_tuples[index]

            img = cv2.imread(str(img_fl), -1)
            mask = cv2.imread(str(mask_fl), -1)

            aug_res = self.augs(image=img.astype(np.uint8), mask=mask)
            img, mask = aug_res.get('image'), aug_res.get('mask')
            img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)

            data_item = tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32), None

        return data_item


def get_data_loaders(data: list or np.ndarray, train_batch_size: int, train_augs, val_augs, test_augs, inf_augs, val_prop: float = .2, logger: logging.Logger = None):
    train_dl = val_dl = test_dl = inf_dl = None
    if train_batch_size > 0 and train_augs is not None and val_augs is not None:
        train_data, val_data = get_train_val_split(data_list=data, val_prop=val_prop, logger=logger)

        # - Create the DataLoader object
        train_dl = DataLoader(
            data_tuples=train_data,
            batch_size=train_batch_size,
            augs=train_augs,
            loader_type='train',
            logger=logger
        )

        if isinstance(val_data, np.ndarray) and val_data.shape[0] > 0:
            val_dl = DataLoader(
                data_tuples=val_data,
                batch_size=VAL_BATCH_SIZE,
                augs=train_augs,
                loader_type='validation',
                logger=logger
            )
    elif test_augs is not None:
        test_dl = DataLoader(
            data_tuples=data,
            batch_size=TEST_BATCH_SIZE,
            augs=test_augs,
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
