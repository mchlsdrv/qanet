import numpy as np
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

from utils.general_utils.aux_funcs import (
    get_runtime,
    decode_file,
    info_log,
)

from utils.image_utils.augmentations import (
    augment,
)

from configs.general_configs import (
    DEBUG_LEVEL,
    PROFILE,
    SHUFFLE_CROPS,
    BATCH_QUEUE_MAXSIZE
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, name: str, data_files: list, batch_size: int, logger: logging.Logger = None):

        # FILES
        self.name = name
        self.img_seg_fls = data_files
        self.n_fls = len(self.img_seg_fls)
        self.logger = logger

        # BATCH QUEUE
        self.btch_q = mlp.Queue(maxsize=BATCH_QUEUE_MAXSIZE)

        # BATCH
        self.btch_sz = batch_size

        # START THE DATA LOADING PROCESS
        self.data_loading_prcs = mlp.Process(target=self._enqueue_batches, args=(self.btch_q,))
        self.data_loading_prcs.start()

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
        if DEBUG_LEVEL:
            info_log(logger=self.logger, message=f'Dequeueing item for {self.name} procedure...')

        return self.btch_q.get()

    def _enqueue_batches(self, batch_queue):
        with tf.device('cpu'):
            while True:
                # - Shuffle files before the next epoch
                np.random.shuffle(self.img_seg_fls)

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

                    btch_fls = self.img_seg_fls[btch_fl_idxs]

                    # - For each file in the batch files
                    t_strt = time.time()
                    btch_fls = list(map(decode_file, btch_fls))
                    # TODO - Bottleneck
                    for idx, (img_fl, seg_fl) in enumerate(btch_fls):

                        # -> Read the image and apply preprocessing functions on it
                        img = preprocess_image(load_image(img_fl))

                        # -> No need to apply the preprocessing on the label, as it consists from a running index
                        seg = load_image(seg_fl)

                        img_crp, seg_crp, aug_seg_crp = augment(image=img, segmentation=seg, logger=self.logger)
                        trgt_seg_msr = get_seg_measure(
                            ground_truth=seg_crp,
                            segmentation=aug_seg_crp
                        )

                        # -> Add the crop into the appropriate container
                        img_crps_btch.append(img_crp)
                        aug_seg_crps_btch.append(aug_seg_crp)
                        trgt_seg_msrs_btch.append(trgt_seg_msr)

                    if PROFILE:
                        info_log(logger=self.logger, message=f'Getting batch of {len(btch_fls)} took {get_runtime(seconds=int(time.time() - t_strt))}')

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
                    batch_queue.put(
                        (
                            (
                                tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
                                tf.convert_to_tensor(aug_seg_crps_btch, dtype=tf.float32)
                            ),
                            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
                        )
                    )

                    if DEBUG_LEVEL:
                        info_log(logger=self.logger, message=f'{self.name} batch #{btch_idx} of shape {img_crps_btch.shape} was successfully loaded to the {self.name} queue! {self.name} queue has {self.btch_q.qsize()} / {BATCH_QUEUE_MAXSIZE} ({100 * self.btch_q.qsize() / BATCH_QUEUE_MAXSIZE}% full) items!')

    def get_queue_load(self, queue: mlp.Queue):
        return self.btch_q.qsize() / BATCH_QUEUE_MAXSIZE

    def stop_data_loading(self):
        self.data_loading_prcs.join()
