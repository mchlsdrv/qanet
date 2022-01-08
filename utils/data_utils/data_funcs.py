import os
import numpy as np
import copy
import pandas as pd
import pathlib
import tensorflow as tf
from functools import partial
from utils.general_utils import aux_funcs
from utils.train_utils import train_funcs
from utils.image_utils import image_funcs
from configs.general_configs import (
    VAL_PROP
)

def configure_shapes(images, labels, shape):
    images.set_shape(shape)
    labels.set_shape([])
    return images, labels


def get_val_ds(dataset, validation_split):
    dataset = dataset.shuffle(buffer_size=1024)
    ds_size = dataset.cardinality().numpy()
    n_val = int(validation_split * ds_size)

    return dataset.take(n_val)


def rename_files(files_dir_path: pathlib.Path):
    # 1) Rename the files to have consequent name
    idx = 1
    for root, folders, files in os.walk(files_dir_path):
        for file in files:
            file_type = file.split('.')[-1]
            os.rename(f'{root}/{file}', f'{root}/{idx}.{file_type}')
            idx += 1


def get_dataset_from_tiff(data_dir_path, input_image_shape, batch_size):
    # 1) Create the global dataset
    # - Rename the fies to have running index as the name
    rename_files(files_dir_path=data_dir_path)

    # 2) Split the dataset into train and validation
    # 2.1) Create the train dataset
    train_ds = tf.data.Dataset.list_files(str(data_dir_path / '*.tiff'))
    train_ds = train_ds.map(lambda x: tf.numpy_function(partial(image_funcs.get_crop, image_shape=input_image_shape, get_label=True), [x], (tf.float32, tf.float32)))
    train_ds = train_ds.map(partial(configure_shapes, shape=input_image_shape))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.shuffle(buffer_size=10*n_samples, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()
    n_samples = train_ds.cardinality().numpy()
    print(f'- Number of train samples: {n_samples}')

    # 2.2) Create the validation dataset
    val_ds = get_val_ds(dataset=train_ds, validation_split=VAL_PROP)
    val_ds = val_ds.map(lambda x: tf.numpy_function(partial(image_funcs.get_crop, image_shape=input_image_shape, get_label=True), [x], (tf.float32, tf.float32)))
    val_ds = val_ds.map(partial(configure_shapes, shape=input_image_shape))
    val_ds = val_ds.batch(4)
    val_ds = val_ds.shuffle(buffer_size=1000)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.repeat()
    n_val = val_ds.cardinality().numpy()

    print(f'- Number of validation samples ({100*validation_split:.2f}%): {n_val}')
    return train_ds, val_ds


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, image_dir: dict, image_shape: tuple, batch_size: int, preprocessing_func, shuffle: bool = True):
        self.image_dir = data_dir.get('image')
        self.ground_truth_dir = data_dir.get('ground_truth')
        self.segmentations_dir = data_dir.get('segmentations')
        self.img_gt_seg_files = self._get_files()
        self.n_files = len(self.img_gt_seg_files)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.preproc_func = preprocessing_func
        self.shuffle = shuffle

    def _get_files(self):
        '''
        The files in the folders are asummed to have last three characters of their name
        an index which should match for all the triplets of (img, gt, seg)
        '''

        img_fls = list()
        for root, _, image_files in os.walk(self.image_dir):
            for image_file in image_files:
                file_name = image_file[:image_file.index('.')].split('/')[-1]
                img_fls.append(f'{root}/{image_file}')

        gt_fls = list()
        for root, _, gt_files in os.walk(self.ground_truth_dir):
            for gt_file in gt_files:
                file_name = gt_file[:gt_file.index('.')].split('/')[-1]
                gt_fls.append(f'{root}/{gt_file}')

        img_gt_seg_files = list()
        for root, _, seg_files in os.walk(self.segmentations_dir):
            for img_fl, gt_fl, seg_file in zip(img_fls, gt_fls, seg_files):

                img_name = img_fl[:img_fl.index('.')].split('/')[-1]
                img_idx = img_name[-3:]

                gt_file_name = gt_fl[:gt_fl.index('.')].split('/')[-1]
                gt_idx = gt_file_name[-3:]

                seg_name = seg_file[:seg_file.index('.')].split('/')[-1]
                seg_idx = seg_name[-3:]

                # -1- If the files match
                if img_idx == gt_idx == seg_idx:
                    img_gt_seg_files.append((img_fl, gt_fl, f'{root}/{seg_file}'))
        return img_gt_seg_files

    def __len__(self):
        return int(np.floor(self.n_files / self.batch_size))

    def __getitem__(self, index):
        if index + self.batch_size <= self.n_files:
            # -1- If there are enough files to fit in the batch
            btch_fls = self.img_gt_seg_files[index:index+self.batch_size]
        else:
            # -2- If there are no batch_size files left
            btch_fls = self.img_gt_seg_files[index:]
        return self._get_batch(batch_files=btch_fls)

    def _get_batch(self, batch_files: list):
        crop_btch = []
        seg_btch = []
        mod_seg_btch = []
        trgt_jaccards = []
        file_names = []

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


