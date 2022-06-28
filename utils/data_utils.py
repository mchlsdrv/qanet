import os
import albumentations as A
import re
import pickle as pkl
import numpy as np
import pathlib
import tensorflow as tf
import logging
from tqdm import tqdm
from copy import copy
from custom.augmentations import (
    train_augmentations,
    validation_augmentations,
    mask_deformations,
)

from utils.image_funcs import (
    get_contours,
    load_image,
)

from utils.aux_funcs import (
    info_log,
    calc_jaccard,
)

from configs.general_configs import (
    TEMP_DIR,
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, name: str, data_files: list, batch_size: int, image_size: int, augmentation_func: A.core.composition.Compose, reload_data: bool = False, reload: bool = True, logger: logging.Logger = None):

        # FILES
        self.name = name
        self.img_seg_fls = data_files
        self.logger = logger
        self.image_size = image_size
        self.augmentation_func = augmentation_func
        self.mask_deformations = mask_deformations()

        tmp_fl = None
        if isinstance(TEMP_DIR, pathlib.Path):
            tmp_fl = TEMP_DIR / f'{self.name}_imgs_segs_temp.npy'

        # - Load the data
        if reload:
            info_log(logger=self.logger, message=f'Loading data file from {self.name} file...')
            if tmp_fl is not None:
                info_log(logger=self.logger, message=f'Saving the temp data to \'{tmp_fl}\'...')
            self.imgs_segs = self._get_data(temp_file=tmp_fl)
        elif tmp_fl.is_file():
            info_log(logger=self.logger, message=f'Loading temp {self.name} data from \'{tmp_fl}\'...')
            self.imgs_segs = np.load(str(tmp_fl))

        # self._clean_blanks()

        self.n_fls = len(self.imgs_segs)

        info_log(logger=self.logger, message=f'Number of clean {self.name} samples: {self.imgs_segs.shape}')

        # BATCH
        # - If the batch size is larger then the number of files - configure it as the number of files
        self.batch_size = batch_size if batch_size <= self.n_fls else self.n_fls

    def __len__(self):
        """
        > Returns the number of batches
        """
        return int(np.floor(self.n_fls / self.batch_size)) if self.batch_size > 0 else 0

    def __getitem__(self, index):
        """
        > Returns a batch of in a form of tuple:
            (image crops, segmentation crops, augmented segmentation crops, target seg measure of the crops)
        """
        # - Get the start index
        btch_idx_strt = index * self.batch_size

        # - Get the end index
        btch_idx_end = btch_idx_strt + self.batch_size - 1

        # -1- If there are enough files to fit in the batch
        # print(f'{btch_idx_end} >? {self.n_fls} = {btch_idx_end >= self.n_fls}')
        if btch_idx_end >= self.n_fls:
            # print('>>>')
            # -2- If there are no batch_size files left
            btch_idx_end = btch_idx_strt + (self.n_fls - btch_idx_strt) - 2

        return self.get_batch(index_start=btch_idx_strt, index_end=btch_idx_end)

    def _clean_blanks(self):
        """
        Cleans all the images there theres no data, i.e., blank or only random noise
        """
        clean_data = []
        for idx, (img, seg) in enumerate(self.imgs_segs):
            bin_seg = copy(seg)

            # - Find the indices of the current labels' class in the segmentation label
            seg_px = np.argwhere(bin_seg > 0)

            # - Separate the indices into x and y coordinates
            seg_pix_xs, seg_pix_ys = seg_px[:, 0], seg_px[:, 1]

            # - Mark the entries at the indices that correspond to the label as '1', to produce a binary label
            bin_seg[(seg_pix_xs, seg_pix_ys)] = 1

            # - Images should be of type UINT8
            bin_seg = bin_seg.astype(np.uint8)

            # - Find the contours in the image
            _, centroids = get_contours(image=bin_seg)

            # - If the image has at least one contour (i.e., it is not blank or noise) - add it to the data
            if centroids:
                clean_data.append((img, seg))

        self.imgs_segs = np.array(clean_data)

    def _get_data(self, temp_file: pathlib.Path = None):
        info_log(logger=self.logger, message=f'Creating temp data...')
        imgs, segs = list(), list()
        for img_fl, seg_fl in tqdm(self.img_seg_fls):
            imgs.append(load_image(img_fl))
            segs.append(load_image(seg_fl))

        info_log(logger=self.logger, message=f'Converting list to array')
        imgs_segs = np.array(list(zip(imgs, segs)))

        # - If the temp_file is specified, the data will be saved to a temp file
        if temp_file is not None:
            info_log(logger=self.logger, message=f'Creating \'temp\' directory, if does not exists')
            if not temp_file.parent.is_dir():
                os.makedirs(temp_file.parent)

            info_log(logger=self.logger, message=f'Saving temp data to \'{temp_file}\'')
            np.save(str(temp_file), imgs_segs)

        return imgs_segs

    @staticmethod
    def _shuffle_crops(images, segmentations, seg_measures):
        rnd_idxs = np.arange(images.shape[0])
        np.random.shuffle(rnd_idxs)

        return images[rnd_idxs], segmentations[rnd_idxs], seg_measures[rnd_idxs]

    def get_batch(self, index_start, index_end):
        img_aug_btch = list()
        seg_aug_btch = list()
        seg_dfrmd_btch = list()
        seg_msrs_btch = list()

        # - Get batch files
        # - Get the file indices for the current batch thread.
        #   * Each thread gets only the indices which he returns
        btch_fl_idxs = np.arange(index_start, index_end)

        btch_fls = self.imgs_segs[btch_fl_idxs]

        # TODO - Bottleneck
        for idx, (img, seg) in enumerate(btch_fls):
            btch_ts = np.array([])

            # - For each file in the batch files
            img, seg = img[:, :, 0], seg[:, :, 0]

            # - Augmentation
            # -> To increase the training set
            res = self.augmentation_func(image=img, mask=seg)
            img_aug, seg_aug = res.get('image'), res.get('mask')

            # -> To produce a pseudo-segmentations to compute the prediction results
            res = self.mask_deformations(image=img_aug, mask=seg_aug)
            seg_dfrmd = res.get('mask')

            # -> Add the image into the appropriate container
            img_aug_btch.append(img_aug)
            seg_aug_btch.append(seg_aug)
            seg_dfrmd_btch.append(seg_dfrmd)

        # - Convert the crops to numpy arrays
        img_aug_btch = np.array(img_aug_btch, dtype=np.float32)
        seg_aug_btch = np.array(seg_aug_btch, dtype=np.float32)
        seg_dfrmd_btch = np.array(seg_dfrmd_btch, dtype=np.float32)

        # - Calculate the seg measure for the current batch
        seg_msrs_btch = calc_jaccard(R=seg_aug_btch, S=seg_dfrmd_btch)

        # - Shuffle batch crops
        img_aug_btch, seg_dfrmd_btch, seg_msrs_btch = self._shuffle_crops(
            images=img_aug_btch,
            segmentations=seg_dfrmd_btch,
            seg_measures=seg_msrs_btch
        )

        return (tf.convert_to_tensor(img_aug_btch, dtype=tf.float32), tf.convert_to_tensor(seg_dfrmd_btch, dtype=tf.float32)), tf.convert_to_tensor(seg_msrs_btch, dtype=tf.float32)


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


def get_files_from_dir(images_dir: str or pathlib.Path, segmentations_dir: str or pathlib.Path):
    img_fls, seg_fls = list(), list()

    for root, dirs, files in os.walk(images_dir):
        for file in files:
            img_fls.append(f'{root}/{file}')

    for root, dirs, files in os.walk(segmentations_dir):
        for file in files:
            seg_fls.append(f'{root}/{file}')

    return list(zip(img_fls, seg_fls))


def get_train_val_split(data: list or np.ndarray, validation_proportion: float = .2, logger: logging.Logger = None):
    n_items = len(data)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * validation_proportion)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data, dtype=object)

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    # - The items for training are the once which are not included in the validation set
    train_data = np_data[np.setdiff1d(item_idxs, val_idxs)]

    info_log(logger=logger, message=f'| Number of train data files : {len(train_data)} | Number of validation data files : {len(val_data)} |')

    return train_data, val_data


def build_metadata(data_dir: str or pathlib.Path, shape: tuple, min_val: int or float = 0.0, max_val: int or float = 255.0):
    """
    Build a meta data file for a multiple directory data format.
    Expects the files in the following format under the data_dir:
        - Image files for each class in a directory 0x
        - Segmentations for each image in a directory 0x_XXXX_XXXX
    *  x - should be an integer
    ** X - may be anything
    """
    file_dict = {}
    for root, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            for sub_root, _, files in os.walk(f'{root}/{dir}'):
                for file in files:
                    # dir_name = pathlib.Path(root).name
                    if file_dict.get(dir[:2]) is None:
                        file_dict[dir[:2]] = [f'{dir}/{file}']
                    else:
                        file_dict.get(dir[:2]).append(f'{dir}/{file}')

    for dir_name in file_dict:
        # - Get the files in the current dir
        files = file_dict[dir_name]

        # - Sort them by image - segmentation
        files = sorted(sorted(files, key=lambda x: x[-7:]), key=lambda x: x[:2])

        # - Group the files as tuples of the format (image, segmentation)
        files = [[files[i], files[i+1]] for i in range(0, len(files)-1, 2)]

        # - Ensure that the image file will be placed before the corresponding mask file
        files = [sorted(fl_tpl, key=lambda x: len(x)) for fl_tpl in files]

        # - Save the metadata file
        pkl.dump(dict(filelist=files, max=max_val, min=min_val, shape=shape), pathlib.Path(f'{data_dir}/metadata_{dir_name}.pkl').open(mode='wb'))


def get_data_files(data_dir: str, metadata_configs: dict, validation_proportion: float = .2, logger: logging.Logger = None):

    build_metadata(data_dir=data_dir, shape=metadata_configs.get('shape'), min_val=metadata_configs.get('min_val'), max_val=metadata_configs.get('max_val'))

    train_fls, val_fls = get_train_val_split(
        data=get_files_from_metadata(
            root_dir=data_dir,
            metadata_files_regex=metadata_configs.get('regex'),
            logger=logger
        ),
        validation_proportion=validation_proportion,
        logger=logger
    )
    return train_fls, val_fls


def get_data_loaders(data_dir: str or pathlib.Path,  metadata_configs: dict, split_proportion: float, batch_size: int, image_size: int, configs: dict, reload_data: bool = False, logger: logging.Logger = None):
    train_fls, val_fls = get_data_files(
        data_dir=data_dir,
        metadata_configs=metadata_configs,
        validation_proportion=split_proportion,
        logger=logger
    )

    # - Create the DataLoader object
    train_dl = DataLoader(
        name='train',
        data_files=train_fls,
        batch_size=batch_size,
        image_size=image_size,
        augmentation_func=train_augmentations(configs=configs.get('augmentation_configs')),
        reload_data=reload_data,
        logger=logger
    )

    val_dl = DataLoader(
        name='val',
        data_files=val_fls,
        batch_size=batch_size,
        image_size=image_size,
        augmentation_func=validation_augmentations(configs=configs.get('augmentation_configs')),
        reload_data=reload_data,
        logger=logger
    )

    return train_dl, val_dl
