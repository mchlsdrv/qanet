import io
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging

from utils import augs

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


def get_random_mask(masks_root: pathlib.Path or str,
                    image_file: pathlib.Path or str):
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
    seg_scr = str_2_float(str_val=seg_scr_str if '-' not in seg_scr_str else
                          seg_scr_str[:seg_scr_str.index('-')])

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
    def __init__(self, mode: str, data_dict: dict, file_keys: list,
                 crop_height: int, crop_width: int, batch_size: int,
                 calculate_seg_score: bool = True,
                 masks_dir: pathlib.Path or str = None, logger: logging = None):
        self.mode = mode
        self.calc_seg_score = calculate_seg_score

        self.data_dict = data_dict
        self.file_keys = np.array(file_keys, dtype=object)

        # - Ensure the batch_size is positive
        self.batch_size = batch_size if batch_size > 0 else 1

        self.train_augs = augs.train_augs(crop_height=crop_height,
                                          crop_width=crop_width)
        self.inf_augs = augs.inference_augs(crop_height=crop_height,
                                            crop_width=crop_width)
        self.mask_augs = augs.mask_augs(image_width=crop_width)

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

        if self.mode == 'training':
            item = self.get_batch_fast_mode(start_index=start_idx,
                                            end_index=end_idx)
        elif self.mode == 'inference':
            item = self.get_batch_inference_mode(index=start_idx)
        elif self.mode == 'test':
            item = self.get_batch_test_mode(index=start_idx)
        else:
            err_log(logger=self.logger,
                    message=f'\'{self.mode}\' mode requires the masks_dir '
                            f'argument to point to a valid location '
                            f'({self.masks_dir}) and be of type '
                            f'\'pathlib.Path\', but is of type '
                            f'\'{type(self.masks_dir)}\' ')

        return item

    def get_batch_fast_mode(self, start_index, end_index):
        t_strt = time.time()

        btch_imgs_aug = []
        btch_msks_gt = []
        btch_msks_aug = []
        btch_seg_scrs = []

        for img_fl in self.file_keys[start_index:end_index]:
            # <1> Get the image and the mask
            img, _, msk_gt = self.data_dict.get(img_fl)
            img = img[..., -1]
            msk_gt = msk_gt[..., -1]

            # <2> Get the random augmentation and the corresponding seg measure
            msk, seg_scr = get_random_mask(masks_root=self.masks_dir,
                                           image_file=img_fl)
            msk = msk[..., -1]

            # <3> Perform the image transformations
            aug_res = self.train_augs(image=img, mask=msk)
            img, msk = aug_res.get('image'), aug_res.get('mask')

            # <4> Apply image transformations
            img = transform_image(image=img)

            # - Add the data to the corresponding lists
            btch_imgs_aug.append(img)
            btch_msks_gt.append(msk_gt)
            btch_msks_aug.append(msk)
            btch_seg_scrs.append(seg_scr)

        # - Convert to tensors

        # - Images
        btch_imgs_aug = tf.convert_to_tensor(np.array(btch_imgs_aug),
                                             dtype=tf.float64)

        # - Seg measures
        btch_seg_scrs = tf.convert_to_tensor(btch_seg_scrs, dtype=tf.float64)
        btch_msks_aug = np.array(btch_msks_aug)
        if self.calc_seg_score:
            btch_seg_scrs = calc_seg_score(gt_masks=np.array(btch_msks_gt),
                                           pred_masks=btch_msks_aug)

        # - Masks
        btch_msks_aug = instance_2_categorical(masks=btch_msks_aug)
        try:
            btch_msks_aug = tf.convert_to_tensor(btch_msks_aug,
                                                 dtype=tf.float64)
        except Exception as err:
            print(f'''
            =======================================================
            - {err}
            =======================================================
            - type(btch_msks_aug): {type(btch_msks_aug)}
            - btch_msks_aug.shape: {btch_msks_aug.shape}
            =======================================================
            ''')

        return (btch_imgs_aug, btch_msks_aug), btch_seg_scrs

    def get_batch_regular_mode(self, start_index, end_index):
        t_strt = time.time()
        btch_imgs_aug = []
        btch_msks_aug = []
        btch_msks_dfrmd = []
        btch_js = []

        btch_data = self.image_mask_tuples[start_index:end_index, ...]
        btch_imgs, btch_msks = btch_data[:, 0, ...], btch_data[:, 1, ...]
        for img, msk in zip(btch_imgs, btch_msks):
            # <1> Perform the image transformations
            aug_res = self.transforms(image=img.astype(np.uint8),
                                      mask=msk.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # <2> Perform image and mask augmentations
            aug_res = self.image_mask_augs(image=img_aug.astype(np.uint8),
                                           mask=msk_aug.astype(np.uint8))
            img_aug, msk_aug = aug_res.get('image'), aug_res.get('mask')

            # <3> Perform mask augmentations
            msks_aug_res = self.mask_augs(image=img_aug, mask=msk_aug)
            msk_dfrmd = msks_aug_res.get('mask')

            # - Add the data to the corresponding lists
            btch_imgs_aug.append(img_aug)
            btch_msks_aug.append(msk_aug)
            btch_msks_dfrmd.append(msk_dfrmd)

        # - Calculate the seg measure for the batch
        # <1> Convert the btch_msks_aug to numpy to calculate the seg measure
        btch_msks_aug = np.array(btch_msks_aug)
        # <2> Convert the btch_msks_dfrmd to numpy to calculate the seg measure
        btch_msks_dfrmd = np.array(btch_msks_dfrmd)
        # <3> Calculate the seg measure of the aug masks with the GT masks,
        # and convert it to tensor
        btch_js = calc_seg_score(gt_masks=btch_msks_aug,
                                 pred_masks=btch_msks_dfrmd)
        # <4> Convert btch_js to tensor
        btch_js = tf.convert_to_tensor(btch_js, dtype=tf.float16)

        # - Convert the btch_imgs_aug to numpy array and then to tensor
        btch_imgs_aug = tf.convert_to_tensor(np.array(btch_imgs_aug),
                                             dtype=tf.float16)

        # - Convert the btch_masks_aug to tensor right away, as it was already
        # converted to numpy in <1>
        btch_msks_dfrmd = instance_2_categorical(masks=btch_msks_dfrmd)
        btch_msks_dfrmd = tf.convert_to_tensor(btch_msks_dfrmd,
                                               dtype=tf.float16)

        return (btch_imgs_aug, btch_msks_dfrmd), btch_js

    def get_batch_test_mode(self, index):
        # - Get the key of the image
        img_key = self.file_keys[index]

        # - Get the image and the mask
        img, _, msk = self.data_dict.get(img_key)
        img, msk = img.astype(np.uint8), msk.astype(np.uint8)

        # - Apply image transformations
        img = transform_image(image=img)

        # - Augment the image and the mask
        aug_res = self.inf_augs(image=img, mask=msk)
        img, msk = aug_res.get('image'), aug_res.get('mask')

        # - Transform the mask from instance segmentation representation to
        # categorical, i.e., 3 classes - background, inner part and the boundary
        msk = instance_2_categorical(masks=msk)

        img = img[..., -1]
        msk = msk[..., -1]

        # - Convert the image and the mask to  tensor
        img, msk = tf.convert_to_tensor([img], dtype=tf.float64), \
            tf.convert_to_tensor([msk], dtype=tf.float64)

        return img, msk, img_key

    def get_batch_inference_mode(self, index):
        # - Get the key of the image
        img_key = self.file_keys[index]

        # - Get the image and the mask
        img, _, msk = self.data_dict.get(img_key)
        img, msk = img.astype(np.uint8), msk.astype(np.uint8)

        # - Apply image transformations
        img = transform_image(image=img)

        # - Augment the image and the mask
        aug_res = self.inf_augs(image=img, mask=msk)
        img, msk = aug_res.get('image'), aug_res.get('mask')
        msk = instance_2_categorical(masks=msk)

        img = img[..., -1]
        msk = msk[..., -1]

        # - Convert the image and the mask to  tensor
        img, msk = tf.convert_to_tensor([img], dtype=tf.float64), \
            tf.convert_to_tensor([msk], dtype=tf.float64)

        return img, msk, img_key


def get_data_loaders(mode: str, data_dict: dict, hyper_parameters: dict,
                     logger: logging.Logger = None):

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
            mode=mode,
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
