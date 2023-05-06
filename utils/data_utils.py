from functools import partial

import numpy as np
import tensorflow as tf
import logging

from utils import augmentations
from utils.augmentations import elastic_transform

from utils.aux_funcs import (
    get_train_val_split,
    instance_2_categorical,
    err_log,
    calc_seg_score,
    transform_image,
    connect_cells
)

P_ELASTIC = 0.1
P_CONNECT_CELLS = 0.4


def get_data_loaders(mode: str, data_dict: dict, hyper_parameters: dict, logger: logging.Logger = None):
    train_files, val_files = get_train_val_split(
        data_list=list(data_dict.keys()),
        val_prop=hyper_parameters.get('training')['val_prop'], logger=logger
    )

    # - Create the DataLoader object
    train_dl = DataLoader(
        mode=mode,
        data_dict=data_dict,
        file_keys=train_files,
        crop_height=hyper_parameters.get('augmentations')['crop_height'],
        crop_width=hyper_parameters.get('augmentations')['crop_width'],
        batch_size=hyper_parameters.get('training')['batch_size'],
        calculate_seg_score=hyper_parameters.get(
            'data')['image_height'] > hyper_parameters.get(
            'augmentations')['crop_height'] or hyper_parameters.get(
            'data')['image_width'] > hyper_parameters.get(
            'augmentations')['crop_width'],
        logger=logger
    )

    val_dl = None
    if len(val_files) > 0:
        val_dl = DataLoader(
            mode='validation',
            data_dict=data_dict,
            file_keys=val_files,
            crop_height=hyper_parameters.get('augmentations')['crop_height'],
            crop_width=hyper_parameters.get('augmentations')['crop_width'],
            batch_size=hyper_parameters.get('training')['val_batch_size'],
            calculate_seg_score=hyper_parameters.get(
                'data')['image_height'] > hyper_parameters.get(
                'augmentations')['crop_height'] or hyper_parameters.get(
                'data')['image_width'] > hyper_parameters.get(
                'augmentations')['crop_width'],
            logger=logger
        )

    return train_dl, val_dl


class DataLoader(tf.keras.utils.Sequence):
    """
    This object operates in two modes:

    1) Regular Mode:
    The image and its mask are used in an on-the-fly manner, where the mask is
    augmented to simulate a new segmentation, and
    the seg measure is calculated for each new sample batch. This mode is
    relatively slow.
    (*) To use this mode provide masks_dir=None.

    2) Preprocessed Mode:
    The masks should be produced and saved in advance in the following format -
        - a - Root dir containing the generated samples
        - b - All the generated masks for each image file will be placed in a
        new dir under the root,
        where the sub dirs of the file will be separated by '_'
        - c - Generated masks will have the seg measure as their name under the
        directory of the file
    This mode is much faster, as it doesn't require mask augmentation and the
    calculation of the seg measure.
    (*) To use this mode provide the images_root and masks_root arguments are
    valid pathable objects (i.e., pathlib.Path or str)
    containing the path to the
    root_dir in the aforementioned format.
    """

    def __init__(self, mode: str, data_dict: dict, file_keys: list, crop_height: int, crop_width: int, batch_size: int,
                 calculate_seg_score: bool = True, logger: logging = None):
        self.mode = mode
        self.calc_seg_score = calculate_seg_score

        self.data_dict = data_dict
        self.file_keys = np.array(file_keys, dtype=object)

        self.crop_height = crop_height
        self.crop_width = crop_width

        # - Ensure the batch_size is positive
        self.batch_size = batch_size if batch_size > 0 else 1

        self.train_augs = augmentations.train_augs(crop_height=crop_height, crop_width=crop_width)
        self.train_image_augs = augmentations.train_image_augs()
        self.train_mask_augs = augmentations.train_mask_augs()
        self.image_transforms = augmentations.image_transforms()
        self.apply_elastic = partial(elastic_transform, alpha=crop_width * 2, sigma=crop_width * 0.15)

        self.n_images = len(self.file_keys)

        self.logger = logger

    def __len__(self):
        """
        > Returns the number of batches
        """
        data_length = int(np.floor(self.n_images / self.batch_size))

        return data_length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size if \
            start_idx + self.batch_size < self.n_images else self.n_images - 1

        item = None
        if self.mode in ['training', 'validation']:
            item = self.get_batch_train_validation(start_index=start_idx, end_index=end_idx)
        elif self.mode in ['test', 'inference']:
            item = self.get_batch_test_inference(index=start_idx)
        else:
            err_log(logger=self.logger, message=f'Error in \'{self.mode}\' mode')

        # - Shuffle the train file list on the last batch
        if self.mode == 'training' and end_idx == self.n_images - 1:
            np.random.shuffle(self.file_keys)

        return item

    def get_batch_train_validation(self, start_index, end_index):
        btch_imgs_aug = []
        btch_msks_gt = []
        btch_msks_aug = []
        btch_seg_scrs = []

        for img_fl in self.file_keys[start_index:end_index]:
            # <1> Get the image and the mask
            img, _, msk_gt = self.data_dict.get(img_fl)
            img, msk_gt = img[..., -1], msk_gt[..., -1]

            # <1> Perform the general augmentations which are made on both the image and the mask e.g., rotation, flip,
            # random crop etc.
            aug_res = self.train_augs(image=img, mask=msk_gt)
            img, msk_gt = aug_res.get('image'), aug_res.get('mask')

            # <2> Perform the general transformations on the image only
            img = transform_image(image=img, augment=True)

            # <3> Change the GT mask to simulate the imperfect segmentation
            aug_res = self.train_mask_augs(image=img, mask=msk_gt)
            msk = aug_res.get('mask')

            # - Randomly apply elastic transform
            if np.random.rand() > P_ELASTIC:
                msk = self.apply_elastic(msk)

            # - Randomly connect cells
            if np.random.rand() > P_CONNECT_CELLS:
                msk = connect_cells(mask=msk)

            # <4> Calculate the seg score of the corrupt mask with the GT
            seg_scr = calc_seg_score(msk_gt, msk)

            # <5> Add the data to the corresponding lists
            btch_imgs_aug.append(img)
            btch_msks_gt.append(msk_gt)
            btch_msks_aug.append(msk)
            btch_seg_scrs.append(seg_scr)

        # - Convert to tensors

        # > Images
        btch_imgs_aug = tf.convert_to_tensor(np.array(btch_imgs_aug), dtype=tf.float32)

        # > Masks
        btch_msks_aug = np.array(btch_msks_aug)
        btch_msks_aug = instance_2_categorical(masks=btch_msks_aug)
        btch_msks_aug = tf.convert_to_tensor(btch_msks_aug, dtype=tf.float32)

        # > Seg measures
        btch_seg_scrs = tf.convert_to_tensor(np.array(btch_seg_scrs), dtype=tf.float32)

        return (btch_imgs_aug, btch_msks_aug), btch_seg_scrs

    def get_batch_test_inference(self, index):
        # - Get the key of the image
        img_key = self.file_keys[index]

        # - Get the image and the mask
        img, _, msk = self.data_dict.get(img_key)

        # - Perform the general transformations on the image only
        img = transform_image(image=img, augment=False)

        # - Transform the image
        img, msk = img[..., -1], msk[..., -1]  # - Discard the last channel as it is a gray scale image

        # - Transform the mask
        msk = instance_2_categorical(masks=msk)  # - Transform the mask from instance segmentation representation to
        # categorical, i.e., 3 classes - background, inner part and the boundary

        return img, msk, img_key
