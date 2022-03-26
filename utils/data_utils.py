import os
import re
import pickle as pkl
import numpy as np
import pathlib
import multiprocessing as mlp
import time
import tensorflow as tf
import logging

from utils.image_utils.preprocessings import (
    preprocess_image,
)

from utils.image_utils.image_aux import (
    get_seg_measure,
    load_image,
)

from utils.image_utils.augmentations import (
    augment,
)

# from utils.general_utils.aux_funcs import (
from utils.aux_funcs import (
    get_runtime,
    info_log,
)

from configs.general_configs import (
    TEMP_DIR,
    DEBUG_LEVEL,
    PROFILE,
    ZERO_LOW_JACCARDS,
    SHUFFLE_CROPS,
    NON_EMPTY_CROPS,
    ROTATION,
    AFFINE,
    EROSION,
    DILATION,
    OPENING,
    CLOSING,
    ELASTIC
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, name: str, data_files: list, batch_size: int, reload_data: bool = False, logger: logging.Logger = None):

        # FILES
        self.name = name
        self.img_seg_fls = data_files
        self.n_fls = len(self.img_seg_fls)
        self.logger = logger

        tmp_dt_fl = TEMP_DIR / f'{self.name}_imgs_segs_temp.npy'
        # - Load the temporal data
        if tmp_dt_fl.is_file() and not reload_data:
            info_log(logger=self.logger, message=f'Loading temp data from \'{tmp_dt_fl}\'...')
            self.imgs_segs_temp = np.load(str(tmp_dt_fl))
        # - Or produce and save it
        else:
            info_log(logger=self.logger, message=f'Creating the temp data file from the files, and saving at \'{tmp_dt_fl}\'...')
            self.imgs_segs_temp = self._get_temp_data(temp_data_file=tmp_dt_fl)

        # BATCH
        # - If the batch size is larger then the number of files - configure it as the number of files
        self.btch_sz = batch_size if batch_size <= self.n_fls else self.n_fls

        # BATCH QUEUE
        # - The maximal queue size is as the number of batches
        self.btch_q_max_sz = self.__len__()
        self.btch_q = mlp.Queue(maxsize=self.btch_q_max_sz)

    def __len__(self):
        """
        > Returns the number of batches
        """
        return int(np.floor(self.n_fls / self.btch_sz))

    def __getitem__(self, index):
        """
        > Returns a batch of in a form of tuple:
            (image crops, segmentation crops, augmented segmentation crops, target seg measure of the crops)
        """
        if DEBUG_LEVEL > 1:
            info_log(logger=self.logger, message=f'Dequeueing item for {self.name} procedure...')

        return self.btch_q.get()

    def _get_temp_data(self, temp_data_file: pathlib.Path):
        imgs, segs = list(), list()
        for idx, (img_fl, seg_fl) in enumerate(self.img_seg_fls):
            img = preprocess_image(load_image(img_fl))
            seg = load_image(seg_fl)

            imgs.append(img)
            segs.append(seg)

        imgs_segs = np.array(list(zip(imgs, segs)))

        if not temp_data_file.parent.is_dir():
            os.makedirs(temp_data_file.parent)

        np.save(str(temp_data_file), imgs_segs)

        return imgs_segs

    def enqueue_batches(self):
        with tf.device('cpu'):
            while True:
                # - Shuffle files before the next epoch
                np.random.shuffle(self.imgs_segs_temp)

                btch_idx_strt = 0
                for btch_idx in range(self.__len__()):

                    if DEBUG_LEVEL > 1:
                        info_log(logger=self.logger, message=f'{self.name} Preparing batch thread #{btch_idx} ...')

                    img_crps_btch = list()
                    seg_crps_btch = list()
                    aug_seg_crps_btch = list()
                    trgt_seg_msrs_btch = list()

                    # - Get batch files
                    lck_wait_strt = time.time()

                    # - Get the end index
                    btch_idx_end = btch_idx_strt + self.btch_sz - 1
                    # -1- If there are enough files to fit in the batch
                    if btch_idx_end >= self.n_fls:
                        # -2- If there are no btch_sz files left
                        btch_idx_end = btch_idx_strt + (self.n_fls - btch_idx_strt) - 1

                    # - Get the file indices for the current batch thread.
                    #   * Each thread gets only the indices which he returns
                    btch_fl_idxs = np.arange(btch_idx_strt, btch_idx_end)

                    btch_fls = self.imgs_segs_temp[btch_fl_idxs]

                    # TODO - Bottleneck
                    for idx, (img, seg) in enumerate(btch_fls):
                        btch_ts = np.array([])

                        # - For each file in the batch files
                        t_strt = time.time()

                        img_crp, seg_crp, aug_seg_crp = augment(
                            image=img,
                            segmentation=seg,
                            non_empty_crops=NON_EMPTY_CROPS,
                            rotation=ROTATION,
                            affine=AFFINE,
                            erosion=EROSION,
                            dilation=DILATION,
                            opening=OPENING,
                            closing=CLOSING,
                            elastic=ELASTIC,
                            logger=self.logger
                        )
                        trgt_seg_msr = get_seg_measure(
                            ground_truth=seg_crp,
                            segmentation=aug_seg_crp,
                            zero_low_jaccards=ZERO_LOW_JACCARDS
                        )

                        # -> Add the crop into the appropriate container
                        img_crps_btch.append(img_crp)
                        aug_seg_crps_btch.append(aug_seg_crp)
                        trgt_seg_msrs_btch.append(trgt_seg_msr)

                    # - Convert the crops into numpy arrays
                    img_crps_btch = np.array(img_crps_btch, dtype=np.float32)
                    aug_seg_crps_btch = np.array(aug_seg_crps_btch, dtype=np.float32)
                    trgt_seg_msrs_btch = np.array(trgt_seg_msrs_btch, dtype=np.float32)

                    # - Shuffle batch crops
                    if SHUFFLE_CROPS:
                        rnd_idxs = np.arange(img_crps_btch.shape[0])
                        np.random.shuffle(rnd_idxs)

                        img_crps_btch = img_crps_btch[rnd_idxs]
                        aug_seg_crps_btch = aug_seg_crps_btch[rnd_idxs]
                        trgt_seg_msrs_btch = trgt_seg_msrs_btch[rnd_idxs]

                    # - Enqueue the prepared batch
                    if DEBUG_LEVEL > 1:
                        info_log(logger=self.logger, message=f'{self.name} Loading batch #{btch_idx} of shape {img_crps_btch.shape} to the {self.name} queue ...')

                    self.btch_q.put(
                        (
                            (
                                tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
                                tf.convert_to_tensor(aug_seg_crps_btch, dtype=tf.float32)
                            ),
                            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
                        )
                    )

                    if DEBUG_LEVEL > 1:
                        info_log(logger=self.logger, message=f'{self.name} batch #{btch_idx} of shape {img_crps_btch.shape} was successfully loaded to the {self.name} queue! {self.name} queue has {self.btch_q.qsize()} / {self.btch_q_max_sz} ({100 * self.get_queue_load():.2f}% full) items!')

                    btch_ts = np.append(btch_ts, time.time() - t_strt)

                if PROFILE:
                    info_log(logger=self.logger, message=f'Getting batch of {len(btch_fls)} took on average {get_runtime(seconds=btch_ts.mean())}')

                if DEBUG_LEVEL:
                    info_log(logger=self.logger, message=f'{self.name} queue has {self.btch_q.qsize()} / {self.btch_q_max_sz} items ({100 * self.get_queue_load():.2f}% full)!')

    def get_queue_load(self):
        return self.btch_q.qsize() / self.btch_q_max_sz


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


def get_data_files(data_dir: str, segmentations_dir: str = None, metadata_files_regex: str = None, validation_proportion: float = .2, logger: logging.Logger = None):
    if metadata_files_regex is not None:
        train_fls, val_fls = get_train_val_split(
            data=get_files_from_metadata(
                root_dir=data_dir,
                metadata_files_regex=metadata_files_regex,
                logger=logger
            ),
            validation_proportion=validation_proportion,
            logger=logger
        )
    else:
        train_fls, val_fls = get_train_val_split(
            data=get_files_from_dir(
                images_dir=data_dir,
                segmentations_dir=segmentations_dir
            ),
            validation_proportion=validation_proportion,
            logger=logger
        )
    return train_fls, val_fls


def get_data_loaders(main_name: str, side_name: str, data_dir: str or pathlib.Path, segmentations_dir: str or pathlib.Path, metadata_files_regex: str, split_proportion: float, batch_size: int, reload_data: bool = False, logger: logging.Logger = None):
    main_fls, side_fls = get_data_files(
        data_dir=data_dir, #args.test_image_dir if args.data_from_single_dir else args.test_dir,
        segmentations_dir=segmentations_dir, #args.test_seg_dir if args.data_from_single_dir else None,
        metadata_files_regex=metadata_files_regex, #None if args.data_from_single_dir else METADATA_FILES_REGEX,
        validation_proportion=split_proportion, #args.validation_proportion,
        logger=logger
    )

    # - Create the DataLoader object
    main_dl = DataLoader(
        name=main_name, #'TEST',
        data_files=main_fls,
        batch_size=batch_size,
        reload_data=reload_data,
        logger=logger
    )

    side_dl = None
    if side_fls.any():
        side_dl = DataLoader(
            name=side_name,
            data_files=side_fls,
            batch_size=batch_size,
            reload_data=reload_data,
            logger=logger
        )

    return main_dl, side_dl
