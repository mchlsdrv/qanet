import numpy as np
import time
import tensorflow as tf
import logging
from threading import (
    Thread,
    Lock,
    currentThread
)
from queue import Queue
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
    err_log
)

from utils.image_utils.augmentations import (
    augment,
)

from configs.general_configs import (
    DEBUG_LEVEL,
    PROFILE,
    CROP_SIZE,
    BATCH_SIZE,
    SHUFFLE_CROPS,
    BATCH_QUEUE_CAPACITY,
    BATCH_QUEUE_DTYPES,

)


@tf.function
def get_queue_current_capacity(queue: tf.queue.FIFOQueue):
    return tf.cast(queue.size(), tf.float32) / BATCH_QUEUE_CAPACITY


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, name: str, data_files: list, batch_size: int, logger: logging.Logger = None):

        # FILES
        self.name = name
        self.img_seg_fls = data_files
        self.n_fls = len(self.img_seg_fls)
        self.logger = logger
        # QUEUES
        self.btch_coord = tf.train.Coordinator()

        self.btch_q = tf.queue.FIFOQueue(
            capacity=BATCH_QUEUE_CAPACITY,
            dtypes=BATCH_QUEUE_DTYPES,
            # shapes=BATCH_QUEUE_SHAPES,
        )

        # THREADS
        self.btch_thrds = []

        # BATCH
        self.btch_sz = batch_size

        # EPOCH END INDICATOR ARRAY
        self.epch_end = None
        self.epch_end_lck = Lock()
        self.img_seg_fls_lck = Lock()

        # START THE BATCH THREADS
        self.thrd_coord = tf.train.Coordinator()
        self.start_batch_threads()

        # SHUFFLING THREAD
        self.shff_thrd = Thread(target=self._shuffle_files, name=f'shuffling_thread', daemon=True)
        self.shff_thrd.start()


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

        data = self.btch_q.dequeue()
        return (data[0], data[1]), data[2]
        # return self.btch_q.get()#dequeue()

    def _shuffle_files(self):
        # - Shuffle files each epoch
        while True:
            if DEBUG_LEVEL > 2:
                info_log(logger=self.logger, message=f'{self.name} | BEFORE SHUFFLE | epoch_end.sum() == {self.epch_end.sum()} >= {len(self.btch_thrds)} | {self.name} queue length = {self.btch_q.size()} / {BATCH_QUEUE_CAPACITY} ({100 * self.btch_q.size() / BATCH_QUEUE_CAPACITY}% full)')
            # - If all the treads finished
            # self.epch_end_lck.acquire()
            if self.epch_end.sum() >= len(self.btch_thrds):

                if DEBUG_LEVEL > 1:
                    info_log(
                        logger=self.logger,
                        message=f'''
                        ==============================
                        {self.name} | Shuffling | self.epch_end.sum()  = {self.epch_end.sum()}
                        ==============================
                        '''
                    )

                # -> Shuffle the files
                self.img_seg_fls_lck.acquire()
                np.random.shuffle(self.img_seg_fls)
                self.img_seg_fls_lck.release()

                # -> Reset the indicator array
                for btch_th in self.btch_thrds:

                    crr_thrd_idx = int(btch_th.getName().split('_')[-1])

                    if DEBUG_LEVEL > 2:
                        info_log(logger=self.logger, message=f'\t\tBEFORE - self.epch_end[crr_thrd_idx] = {self.epch_end[crr_thrd_idx]}')
                    self.epch_end[crr_thrd_idx] = 0

                    if DEBUG_LEVEL > 2:
                        info_log(logger=self.logger, message=f'\t\tAFTER - self.epch_end[crr_thrd_idx] = {self.epch_end[crr_thrd_idx]}')

                if DEBUG_LEVEL > 2:
                    info_log(logger=self.logger, message=f'{self.name} | AFTER SHUFFLE | epoch_end.sum() == {self.epch_end.sum()}')
            # self.epch_end_lck.release()

    def _enqueue_batch(self, batch_files_indices):
        while True:
            with tf.device('/cpu:0'):
                crr_thrd_idx = int(currentThread().getName().split('_')[-1])
                # - Load a batch only one time
                if DEBUG_LEVEL > 2:
                    info_log(logger=self.logger, message=f'{self.name} | THREAD {crr_thrd_idx} | self.epch_end[crr_thrd_idx] == {self.epch_end[crr_thrd_idx]} | {self.name} queue length = {self.btch_q.size()} / {BATCH_QUEUE_CAPACITY} ({100 * self.btch_q.size() / BATCH_QUEUE_CAPACITY}% full) items')

                if self.epch_end[crr_thrd_idx] == 0:

                    img_crps_btch = list()
                    seg_crps_btch = list()
                    aug_seg_crps_btch = list()
                    trgt_seg_msrs_btch = list()

                    # - Get batch files
                    lck_wait_strt = time.time()
                    if DEBUG_LEVEL > 1:
                        info_log(logger=self.logger, message=f'{self.name} thread #{crr_thrd_idx} - waiting on self.img_seg_lck...')
                    self.img_seg_fls_lck.acquire()
                    if DEBUG_LEVEL > 1:
                        info_log(logger=self.logger, message=f'{self.name} thread #{crr_thrd_idx} - self.img_seg_lck acquired!')
                    if PROFILE:
                        info_log(logger=self.logger, message=f'{self.name} thread #{crr_thrd_idx} - self.img_seg_lck acquired in {get_runtime(seconds=time.time() - lck_wait_strt)}')

                    btch_fls = self.img_seg_fls[batch_files_indices]
                    self.img_seg_fls_lck.release()

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
                        info_log(logger=self.logger, message=f'Getting batch of {len(btch_fls)} took {get_runtime(seconds=time.time() - t_strt)}')

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
                        info_log(logger=self.logger, message=f'{self.name} Loading batch #{crr_thrd_idx} of shape {img_crps_btch.shape} to the {self.name} queue ...')
                    self.btch_q.enqueue(
                        [
                            tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
                            tf.convert_to_tensor(aug_seg_crps_btch, dtype=tf.float32),
                            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
                        ]
                    )

                    if DEBUG_LEVEL:
                        info_log(logger=self.logger, message=f'{self.name} batch #{crr_thrd_idx} of shape {img_crps_btch.shape} was sussessfuly loaded to the {self.name} queue! {self.name} queue has {self.btch_q.size()} / {BATCH_QUEUE_CAPACITY} ({100 * self.btch_q.size() / BATCH_QUEUE_CAPACITY}% full) items!')
                    # - Update the current batch status as finished in the epoch end indicator array, to know when to shuffle files
                    # self.epch_end_lck.acquire()
                    self.epch_end[crr_thrd_idx] = 1
                    # self.epch_end_lck.release()

    def start_batch_threads(self):

        # - The epoch ends when all the batches have been loaded to the queue
        self.epch_end = np.zeros(self.__len__())

        btch_idx_strt = 0
        # - Number of threads is as the number of batches
        for btch_idx in range(self.__len__()):
            if DEBUG_LEVEL > 1:
                info_log(logger=self.logger, message=f'{self.name} Preparing batch thread #{btch_idx} ...')

            # - Get the end index
            btch_idx_end = btch_idx_strt + self.btch_sz - 1
            # -1- If there are enough files to fit in the batch
            if  btch_idx_end >= self.n_fls:
                # -2- If there are no btch_sz files left
                btch_idx_end = btch_idx_strt + (self.n_fls - btch_idx_strt) - 1

            # - Get the file indices for the current batch thread.
            #   * Each thread gets only the indices which he returns
            fl_idxs = np.arange(btch_idx_strt, btch_idx_end)
            btch_thrd = Thread(target=self._enqueue_batch, args=([fl_idxs]), name=f'batch_thread_{btch_idx}', daemon=True)
            self.thrd_coord.register_thread(btch_thrd)

            # - Append the batch thread to the batch threads list
            self.btch_thrds.append(btch_thrd)

            # - Update the batch iמdex
            btch_idx_strt += self.btch_sz

            # - Start the thread
            if DEBUG_LEVEL > 1:
                info_log(logger=self.logger, message=f'{self.name} Starting batch thread #{btch_idx}...')
            btch_thrd.start()
            if DEBUG_LEVEL:
                info_log(logger=self.logger, message=f'{self.name} Batch thread #{btch_idx} started!')

    def stop_threads(self):
        for btch_thrd in self.btch_thrds:
            btch_thrd.join()
        self.shff_thrd.join()

