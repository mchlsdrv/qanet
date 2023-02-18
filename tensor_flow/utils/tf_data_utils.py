import io
import os
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

from utils import augs
from utils.augs import elastic_transform

from utils.aux_funcs import (
    get_train_val_split,
    get_file_name,
    get_parent_dir_name,
    str_2_float,
    load_image,
    assert_pathable,
    str_2_path,
    check_pathable,
    get_files_under_dir,
    instance_2_categorical,
    err_log,
    calc_seg_score,
    transform_image,
)


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
        masks_dir=hyper_parameters.get('training')['mask_dir'],
        logger=logger
    )

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
            masks_dir=hyper_parameters.get('training')['mask_dir'],
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


def get_random_mask(masks_root: pathlib.Path or str, image_file: pathlib.Path or str):
    # - Assert the mask directory and the image file represent a path
    assert_pathable(argument=masks_root, argument_name='masks_root')
    assert_pathable(argument=image_file, argument_name='image_file')

    # - Make sure the mask directory and the image file are in the pathlib.Path
    # format
    masks_root = str_2_path(path=masks_root)
    image_file = str_2_path(path=image_file)

    # - Get file name
    fl_name = get_file_name(path=image_file)

    # - Load the gt mask
    fl_msk_dir = masks_root / f'{get_parent_dir_name(path=image_file)}' \
                              f'_{fl_name}'

    # - Get the masks which correspond to the current file
    msk_names = np.array(os.listdir(fl_msk_dir), dtype=object)
    n_msks = len(msk_names)

    # - Choose randomly n_samples masks
    rnd_idx = np.random.randint(0, n_msks)
    rnd_msk_name = msk_names[rnd_idx]

    # - Get the name of the mask
    rnd_msk_fl = fl_msk_dir / rnd_msk_name

    # - Get the seg measure of the chosen mask, which should be the name of the
    # file with '_' instead of '.'
    seg_scr_str = get_file_name(path=rnd_msk_fl)

    # - Strip the '-' character in case it was added due to overlap in
    # generated J values
    seg_scr = str_2_float(str_val=seg_scr_str if '-' not in seg_scr_str else seg_scr_str[:seg_scr_str.index('-')])

    # - Load the mask
    msk = load_image(image_file=str(rnd_msk_fl), add_channels=True)

    # - Return the random mask and the corresponding seg measure
    return msk, seg_scr


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
                 calculate_seg_score: bool = True, masks_dir: pathlib.Path or str = None, logger: logging = None):
        self.mode = mode
        self.calc_seg_score = calculate_seg_score

        self.data_dict = data_dict
        self.file_keys = np.array(file_keys, dtype=object)

        # - Ensure the batch_size is positive
        self.batch_size = batch_size if batch_size > 0 else 1

        self.train_augs = augs.train_augs(crop_height=crop_height, crop_width=crop_width)
        self.train_mask_augs = augs.mask_augs()
        self.inf_augs = augs.inference_augs(crop_height=crop_height, crop_width=crop_width)
        self.apply_elastic = partial(elastic_transform, alpha=crop_width * 2, sigma=crop_width * 0.15)

        # - In case the masks are made in advance, in which case there is no
        # need for the self.mask_augs
        self.masks_dir = str_2_path(path=masks_dir)
        self.n_masks = 0 if not check_pathable(path=self.masks_dir) else \
            len(get_files_under_dir(dir_path=self.masks_dir))

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

        if self.mode == 'training' or 'validation':
            item = self.get_batch_train(start_index=start_idx, end_index=end_idx)
        elif self.mode == 'inference':
            item = self.get_batch_inference(index=start_idx)
        elif self.mode == 'test':
            item = self.get_batch_test(index=start_idx)
        else:
            err_log(logger=self.logger,
                    message=f'\'{self.mode}\' mode requires the masks_dir '
                            f'argument to point to a valid location '
                            f'({self.masks_dir}) and be of type '
                            f'\'pathlib.Path\', but is of type '
                            f'\'{type(self.masks_dir)}\' ')

        # - Shuffle the train file list on the last batch
        if self.mode == 'training' and end_idx == self.n_images - 1:
            np.random.shuffle(self.file_keys)

        return item

    def get_batch_train(self, start_index, end_index):
        btch_imgs_aug = []
        btch_msks_gt = []
        btch_msks_aug = []
        btch_seg_scrs = []

        for img_fl in self.file_keys[start_index:end_index]:
            # <1> Get the image and the mask
            img, _, msk_gt = self.data_dict.get(img_fl)
            img = img[..., -1]
            img = transform_image(image=img)

            msk_gt = msk_gt[..., -1]

            # <2> Perform the general augmentations which are made on both the image and the mask e.g., rotation, flip,
            # random crop etc.
            aug_res = self.train_augs(image=img, mask=msk_gt)
            img = aug_res.get('image')
            msk_gt = aug_res.get('mask')

            # <3> Change the GT mask to simulate the imperfect segmentation
            aug_res = self.train_mask_augs(image=img, mask=msk_gt)
            msk = aug_res.get('mask')
            if np.random.rand() > 0.1:
                msk = self.apply_elastic(msk)

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

    def get_batch_test(self, index):
        # - Get the key of the image
        img_key = self.file_keys[index]

        # - Get the image and the mask
        img, _, msk = self.data_dict.get(img_key)

        # - Discard the last channel as it is a gray scale image
        img, msk = img[..., -1], msk[..., -1]

        # - Apply image transformations
        img = transform_image(image=img)

        # - Augment the image and the mask
        aug_res = self.inf_augs(image=img, mask=msk)
        img, msk = aug_res.get('image'), aug_res.get('mask')

        # - Transform the mask from instance segmentation representation to
        # categorical, i.e., 3 classes - background, inner part and the boundary
        msk = instance_2_categorical(masks=msk)

        # - Convert the image and the mask to  tensor
        img, msk = tf.convert_to_tensor([img], dtype=tf.float32), tf.convert_to_tensor([msk], dtype=tf.float32)

        return img, msk, img_key

    def get_batch_inference(self, index):
        # - Get the key of the image
        img_key = self.file_keys[index]

        # - Get the image and the mask
        img, _, msk = self.data_dict.get(img_key)
        # img, msk = img.astype(np.uint8), msk.astype(np.uint8)

        # - Apply image transformations
        img = transform_image(image=img)

        # - Augment the image and the mask
        aug_res = self.inf_augs(image=img, mask=msk)
        img, msk = aug_res.get('image'), aug_res.get('mask')
        msk = instance_2_categorical(masks=msk)

        # - Discard the last channel as it is a gray scale image
        img, msk = img[..., -1], msk[..., -1]

        # - Convert the image and the mask to  tensor
        img, msk = tf.convert_to_tensor([img], dtype=tf.float32), \
            tf.convert_to_tensor([msk], dtype=tf.float32)

        return img, msk, img_key
