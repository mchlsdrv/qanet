import os
import numpy as np
import copy
import pandas as pd
import pathlib
import tensorflow as tf
from collections.abc import Callable as function
from functools import partial
os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
from utils.general_utils import aux_funcs
from utils.train_utils import train_funcs
from utils.image_utils import (
    image_funcs,
    augmentation_funcs,
    preprocessing_funcs
)
from configs.general_configs import (
    VAL_PROP
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data: dict, crop_shape: tuple, batch_size: int, preprocessings: function, augmentations: function, shuffle: bool = True):
        # FILES
        self.img_dir = data.get('images_dir')
        self.seg_dir = data.get('segmentations_dir')
        self.img_seg_fls = self._get_img_seg_files()
        self.n_files = len(self.img_seg_files)
        # SHAPE
        self.image_shape = crop_shape
        # BATCH
        self.batch_size = batch_size
        # FUNCTIONS
        self.preproc = preprocessings
        self.augs = augmentations
        # TRAINING
        self.shuffle = shuffle

    def _get_img_seg_files(self):
        '''
        > This private method pairs between the image files and their corresponding segmentations.
        The files in the folders are asummed to have last three characters of their name
        an index which should match for all the triplets of (img, seg)
        '''

        # - All the image files
        img_fls = list()
        for root, _, image_files in os.walk(self.img_dir):
            for image_file in image_files:
                file_name = image_file[:image_file.index('.')].split('/')[-1]
                img_fls.append(f'{root}/{image_file}')

        # - All the segmentation files
        seg_fls = list()
        for root, _, seg_files in os.walk(self.seg_dir):
            for seg_file in seg_files:
                file_name = seg_file[:seg_file.index('.')].split('/')[-1]
                seg_fls.append(f'{root}/{seg_file}')

        # - Pair the image files to their segmentation files
        img_seg_fls = list()
        for img_fl, seg_fl in zip(img_fls, seg_fls):

            img_name = img_fl[:img_fl.index('.')].split('/')[-1]
            img_idx = img_name[-3:]  # => The last 3 characters are the image index

            seg_name = seg_fl[:seg_fl.index('.')].split('/')[-1]
            seg_idx = seg_name[-3:]  # => The last 3 characters are the image index

            # -1- If the file indices match
            if img_idx == seg_idx:
                img_gt_seg_fls.append((img_fl, seg_fl))

        return img_seg_fls

    def __len__(self):
        '''
        > Returns the number of batches
        '''
        return int(np.floor(self.n_files / self.batch_size))

    def __getitem__(self, index):
        '''
        > Returns a batch of in a form of tuple:
            (image crops, segmentation crops, augmented segmentation crops, target seg measure of the crops)
        '''
        if index + self.batch_size <= self.n_files:
            # -1- If there are enough files to fit in the batch
            btch_fls = self.img_gt_seg_files[index:index+self.batch_size]
        else:
            # -2- If there are no batch_size files left
            btch_fls = self.img_gt_seg_files[index:]
        return self._get_batch(batch_files=btch_fls)

    def _get_batch(self, batch_files: list):
        img_btch = []
        seg_btch = []
        mod_seg_btch = []
        trgt_j= []

        # I) For each file in the files chosen for it
        batch_files = list(map(aux_funcs.decode_file, batch_files))
        for idx, (img_fl, gt_fl, seg_fl) in enumerate(batch_files):
            # 2) Read the image
            img = self.preproc_fun(cv2.imread(img_fl, cv2.IMREAD_UNCHANGED))
            gt = self.preproc_fun(cv2.imread(gt_fl, cv2.IMREAD_UNCHANGED))
            seg = self.preproc_fun(cv2.imread(seg_fl, cv2.IMREAD_UNCHANGED))

            # 3) Apply the preprocessing function
            img = preprocessing_funcs.clahe_filter(image=img)
            img_crp, gt_crp, seg_crp = image_funcs.get_random_crop(
                image_files=[img_fl, gt_fl, seg_fl],
                image_shape=self.image_shape,
            )
            # - Add the original file to the batch
            D_btch.append(X)

            # II) Add each of the neighbors of the original image (first ne)
            # - If the number of the neighbors is greater than 1
            if self.k > 1:
                N_X_files = list(copy.deepcopy(knn_batch_df.loc[X_file_idx, 'neighbors'])[0])
                # - The  image at index 0 is the original image
                N_X_files.pop(0)
                N_X = []
                for N_X_file in N_X_files:
                    ngbr, _ = image_funcs.get_crop(
                        image_file=N_X_file,
                        image_shape=self.image_shape,
                        get_label=False
                    )
                    # - Collect the neighbors of the original image
                    N_X.append(ngbr)
                # - Add the collected neighbors to the neighbors batch
                N_btch.append(N_X)
            else:
                # - If we are interensted only in the closest neighbor
                ngbr, _ = image_funcs.get_crop(
                    image_file=knn_batch_df.loc[X_file_idx, 'neighbors'][0][1],
                    image_shape=self.image_shape,
                    get_label=False
                )
                # - Add the closest neighbor to the neighbors batch
                N_btch.append(ngbr)

        D_btch = np.array(D_btch, dtype=np.float32)
        N_btch = np.array(N_btch, dtype=np.float32)

        if self.shuffle:
            random_idxs = np.arange(D_btch.shape[0])
            np.random.shuffle(random_idxs)
            D_btch = D_btch[random_idxs]
            N_btch = N_btch[random_idxs]
        return tf.convert_to_tensor(D_btch, dtype=tf.float32), tf.convert_to_tensor(N_btch, dtype=tf.float32)


DATA_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/Silver_GT/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
GT_DIR = DATA_DIR / '01_ST/SEG'

if __name__=='__main__':
    data = dict(
        images_dir=IMAGE_DIR,
        segmentations_dir=GT_DIR
    )
    dl = DataLoader(
        data=data,
        image_shape=(254, 254, 1),
        batch_size=32,
        preprocessings=preprocessing_funcs.preprocessings,
        augmentations=augmentation_funcs.augmentations,
        shuffle=True
    )

