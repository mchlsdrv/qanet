import datetime
import os
import pathlib
import argparse
import logging.config
import pickle as pkl
import re
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import cv2
import yaml

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from global_configs.general_configs import (
    OUTPUT_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IN_CHANNELS,
    OUT_CHANNELS,
    EPOCHS,
    TRAIN_BATCH_SIZE,
    VAL_PROP,
    OPTIMIZER_LR,
    OPTIMIZER_LR_DECAY,
    OPTIMIZER_BETA_1,
    OPTIMIZER_BETA_2,
    OPTIMIZER_RHO,
    OPTIMIZER_WEIGHT_DECAY,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_DAMPENING,
    OPTIMIZER_MOMENTUM_DECAY,
    OPTIMIZER,
    DROP_BLOCK_KEEP_PROB,
    DROP_BLOCK_BLOCK_SIZE,
    KERNEL_REGULARIZER_TYPE,
    KERNEL_REGULARIZER_L1,
    KERNEL_REGULARIZER_L2,
    KERNEL_REGULARIZER_FACTOR,
    KERNEL_REGULARIZER_MODE,
    ACTIVATION,
    ACTIVATION_RELU_MAX_VALUE,
    ACTIVATION_RELU_NEGATIVE_SLOPE,
    ACTIVATION_RELU_THRESHOLD,
    ACTIVATION_LEAKY_RELU_ALPHA,
    INFERENCE_DATA_DIR,
    MIN_CELL_PIXELS,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR
)

from pytorch.configs import general_configs as tr_configs
from tensor_flow.configs import general_configs as tf_configs

mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by the wrong thread" exception
plt.style.use('seaborn')  # <= using the seaborn plot style

sns.set()
RC = {
    'font.size': 32,
    'axes.labelsize': 50,
    'legend.fontsize': 30.0,
    'axes.titlesize': 32,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40
}
sns.set_context(rc=RC)


def get_range(value, ranges: np.ndarray):
    for idx, (rng_min, rng_max) in enumerate(ranges):
        if rng_min < value < rng_max:
            break
    return rng_min, rng_max, idx


def str_2_float(str_val: str):
    return float(str_val.replace('_', '.'))


def float_2_str(float_val: float):
    return f'{float_val:.3f}'.replace('.', '_')


def get_file_name(path: str):
    file_name = path[::-1][path[::-1].index('.')+1:path[::-1].index('/')][::-1]
    return file_name


def get_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_contours(image: np.ndarray):
    # - Find the contours
    contours, hierarchies = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # - Find the centroids of the contours
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    return contours, centroids


def get_crop(image: np.ndarray, x: int, y: int, crop_shape: tuple):
    return image[x:x + crop_shape[0], y:y + crop_shape[1]]


def add_channels_dim(image: np.ndarray):
    # - If the image is 2D - add the channel dimension
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=-1)
    return image


def load_image(image_file, add_channels: bool = False):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if add_channels and len(img.shape) < 3:
        img = add_channels_dim(image=img)
    return img


def show_images(images, labels, suptitle='', figsize=(25, 10), save_file: pathlib.Path or str = None, verbose: bool = False, logger: logging.Logger = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=pathlib.Path(save_file), verbose=verbose, logger=logger)


def line_plot(x: list or np.ndarray, ys: list or np.ndarray, suptitle: str, labels: list, colors: tuple = ('r', 'g', 'b'), save_file: pathlib.Path or str = None, logger: logging.Logger = None):
    # assert len(ys) == len(labels) == len(colors), f'The number of data items ({len(ys)}) does not match the number of labels ({len(labels)}) or the number of colors ({len(colors)})!'

    fig, ax = plt.subplots()
    for y, lbl, clr in zip(ys, labels, colors):
        ax.plot(x, y, color=clr, label=lbl)

    plt.legend()

    try:
        save_figure(figure=fig, save_file=save_file, logger=logger)
    except Exception as err:
        err_log(logger=logger, message=f'{err}')


def hit_rate_plot(true: np.ndarray, pred: np.ndarray, save_file: pathlib.Path or str = None, logger: logging.Logger = None):
    # sns.set(font_scale=1)
    abs_err = np.abs(true - pred)
    abs_err_hist, abs_err = np.histogram(abs_err, bins=100, range=(0., 1.))
    abs_err_prob = abs_err_hist / np.sum(abs_err_hist)

    fig, ax = plt.subplots()
    ax.plot(abs_err[:-1], np.cumsum(abs_err_prob), linewidth=2)
    ax.set(xlabel='Absolute Error Tolerance', xlim=(0, 1), xticks=np.arange(0.0, 1.0, 0.2), ylabel='Hit Rate', ylim=(0, 1), yticks=np.arange(0.0, 1.0, 0.2))
    save_dir = pathlib.Path(save_file).parent
    if not save_dir.is_dir():
        os.makedirs(save_dir)
    plt.savefig(str(save_file))
    plt.close()

    return abs_err_hist, abs_err


def absolute_error_plot(true: np.ndarray, pred: np.ndarray, save_file: pathlib.Path or str = None, logger: logging.Logger = None):
    # sns.set(font_scale=1.5)
    abs_err = np.abs(true - pred)
    err_hist, abs_err_bins = np.histogram(abs_err, bins=100, range=(0, 1))

    D = pd.DataFrame({'Absolute Error': abs_err_bins[:-1], 'Frequency': err_hist})
    g = sns.jointplot(x='Absolute Error', y='Frequency', data=D, height=15, space=0.02)

    try:
        save_dir = pathlib.Path(save_file).parent
        if not save_dir.is_dir():
            os.makedirs(save_dir)
        save_figure(figure=g.figure, save_file=save_file, logger=logger)
    except Exception as err:
        err_log(logger=logger, message=f'{err}')
    return err_hist, abs_err_bins


def scatter_plot(x: np.ndarray, y: np.ndarray, save_file: pathlib.Path or str = None, logger: logging.Logger = None):
    # sns.set(font_scale=1.5)

    D = pd.DataFrame({'GT Quality Value': x, 'Estimated Quality Value': y})
    g = sns.jointplot(
        x='GT Quality Value',
        y='Estimated Quality Value',
        marker='o',
        joint_kws={
            'scatter_kws': {
                'alpha': 0.3,
                's': 150
            }
        },
        data=D,
        height=15,
        space=0.02,
        kind='reg'
    )

    try:
        save_dir = pathlib.Path(save_file).parent
        if not save_dir.is_dir():
            os.makedirs(save_dir)
        save_figure(figure=g.figure, save_file=save_file, logger=logger)
    except Exception as err:
        err_log(logger=logger, message=f'{err}')


def str_2_path(path: str):
    return pathlib.Path(path) if isinstance(path, str) else path


def save_figure(figure, save_file: pathlib.Path or str, verbose: bool = False, logger: logging.Logger = None):
    save_file = str_2_path(path=save_file)

    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        figure.savefig(str(save_file))
        plt.close(figure)
        if verbose:
            info_log(logger=logger, message=f'Figure was saved to \'{save_file}\'')


def plot_seg_measure_histogram(seg_measures: np.ndarray, bin_width: float = .1, figsize: tuple = (25, 10), density: bool = False, save_file: pathlib.Path = None):
    vals, bins = np.histogram(seg_measures, bins=np.arange(0., 1. + bin_width, bin_width))
    if density:
        vals = vals / vals.sum()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bins[:-1], vals, width=bin_width, align='edge')
    ax.set(xlim=(0, 1), xticks=np.arange(0., 1.1, .1), xlabel='E[SM] (Mean Seg Measure)', ylabel='P(E[SM])')

    save_figure(figure=fig, save_file=save_file)


def plot_image_histogram(images: np.ndarray, labels: list, n_bins: int = 256, figsize: tuple = (25, 50), density: bool = False, save_file: pathlib.Path = None):
    fig, ax = plt.subplots(2, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):

        vals, bins = np.histogram(img, n_bins, density=True)
        if density:
            vals = vals / vals.sum()
        vals, bins = vals[1:], bins[1:][:-1]  # don't include the 0

        # - If there is only a single plot - no second dimension will be available, and it will result in an error
        if len(images) > 1:
            hist_ax = ax[0, idx]
            img_ax = ax[1, idx]
        else:
            hist_ax = ax[0]
            img_ax = ax[1]

        # - Plot the histogram
        hist_ax.bar(bins, vals)
        hist_ax.set_title('Intensity Histogram')
        max_val = 255 if img.max() > 1 else 1
        hist_ax.set(xlim=(0, max_val), ylim=(0., 1.), yticks=np.arange(0., 1.1, .1), xlabel='I (Intensity)', ylabel='P(I)')

        # - Show the image
        img_ax.imshow(img, cmap='gray')
        img_ax.set_title(lbl)

    save_figure(figure=fig, save_file=save_file)


def decode_file(file):
    if isinstance(file, bytes):
        file = file.decode('utf-8')
    return file


def check_dir(path: str or pathlib.Path):
    dir_valid = False

    # - Convert path to pathlib.Path object

    dir_path = str_2_path(path=path)

    # - Check if file exists
    if dir_path.is_file():
        dir_valid = True

    return dir_valid


def check_file(file_path: str or pathlib.Path):
    path_valid = False

    # - Convert path to pathlib.Path object
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    # - Check if file exists
    if file_path.is_file():
        path_valid = True

    return path_valid


def read_yaml(data_file: pathlib.Path):
    data = None
    if data_file.is_file():
        with data_file.open(mode='r') as f:
            data = yaml.safe_load(f.read())
    return data


def get_train_val_split(data_list: list or np.ndarray, val_prop: float = .2, logger: logging.Logger = None):
    n_items = len(data_list)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * val_prop)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data_list, dtype=object)

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    # - The items for training are the once which are not included in the validation set
    train_data = np_data[np.setdiff1d(item_idxs, val_idxs)]

    info_log(logger=logger, message=f'| Number of train data files : {len(train_data)} | Number of validation data files : {len(val_data)} |')

    return train_data, val_data


def get_logger(configs_file, save_file):
    logger = None
    try:
        if configs_file.is_file():
            with configs_file.open(mode='r') as f:
                configs = yaml.safe_load(f.read())

                # Assign a valid path to the log file
                configs['handlers']['logfile']['filename'] = str(save_file)
                logging.config.dictConfig(configs)

        logger = logging.getLogger(__name__)
    except Exception as err:
        err_log(logger=logger, message=str(err))

    return logger


def info_log(logger: logging.Logger, message: str):
    if isinstance(logger, logging.Logger):
        logger.info(message)
    else:
        print(message)


def warning_log(logger: logging.Logger, message: str):
    if isinstance(logger, logging.Logger):
        logger.warning(message)
    else:
        print(message)


def err_log(logger: logging.Logger, message: str):
    if isinstance(logger, logging.Logger):
        logger.exception(message)
    else:
        print(message)


def data_not_found_err(data_file: str or pathlib.Path, logger: logging.Logger):
    err_log(logger=logger, message=f'Could not load data from \'{data_file}\'!')


def build_metadata(data_dir: str or pathlib.Path, shape: tuple, min_val: int or float = 0.0, max_val: int or float = 255.0):
    """
    Build a metadata file for a multiple directory data format.
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
        files = [[files[i], files[i + 1]] for i in range(0, len(files) - 1, 2)]

        # - Ensure that the image file will be placed before the corresponding mask file
        files = [sorted(fl_tpl, key=lambda x: len(x)) for fl_tpl in files]

        # - Save the metadata file
        pkl.dump(dict(filelist=files, max=max_val, min=min_val, shape=shape), pathlib.Path(f'{data_dir}/metadata_{dir_name}.pkl').open(mode='wb'))


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


def get_data_files(data_dir: str, metadata_configs: dict, val_prop: float = .2, logger: logging.Logger = None):
    build_metadata(data_dir=data_dir, shape=metadata_configs.get('shape'), min_val=metadata_configs.get('min_val'), max_val=metadata_configs.get('max_val'))

    train_fls, val_fls = get_train_val_split(
        data_list=get_files_from_metadata(
            root_dir=data_dir,
            metadata_files_regex=metadata_configs.get('regex'),
            logger=logger
        ),
        val_prop=val_prop,
        logger=logger
    )
    return train_fls, val_fls


def get_model_configs(configs_file: pathlib.Path, logger: logging.Logger):
    model_configs = None
    if configs_file.is_file():
        model_configs = read_yaml(configs_file)
        if model_configs is not None:
            info_log(logger=logger, message=f'The model configs were successfully loaded from \'{configs_file}\'!')
        else:
            info_log(logger=logger, message=f'The model configs file ({configs_file}) does not contain valid model configurations!')
    else:
        info_log(logger=logger, message=f'The model configs file ({configs_file}) does not exist!')
    return model_configs


def scan_files(root_dir: pathlib.Path or str, seg_dir_postfix: str, image_prefix: str, seg_prefix: str, seg_sub_dir: str = None):
    file_tuples = list()
    for root, dirs, _ in os.walk(root_dir):
        root = pathlib.Path(root)
        for dir in dirs:
            seg_dir = root / f'{dir}_{seg_dir_postfix}'
            if seg_dir.is_dir():
                for sub_root, _, files in os.walk(root / dir):
                    for file in files:
                        img_fl = pathlib.Path(f'{sub_root}/{file}')
                        if seg_sub_dir is not None:
                            seg_fl = pathlib.Path(f'{seg_dir}/{seg_sub_dir}/{file.replace(image_prefix, seg_prefix)}')
                        else:
                            seg_fl = pathlib.Path(f'{seg_dir}/{file.replace(image_prefix, seg_prefix)}')
                        if img_fl.is_file() and seg_fl.is_file():
                            file_tuples.append((img_fl, seg_fl))
    return file_tuples


def load_images_from_tuple_list(data_file_tuples: list):
    imgs = []
    msks = []

    print('''
    ====================================================
    == Loading images and corresponding segmentations ==
    ====================================================
    ''')
    files_pbar = tqdm(data_file_tuples)
    for img_fl, msk_fl in files_pbar:
        img = cv2.imread(str(img_fl), -1).astype(np.uint8)
        # - Add the channel dim
        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=-1)

        msk = cv2.imread(str(msk_fl), -1).astype(np.uint8)
        # - Add the channel dim
        if len(msk.shape) < 3:
            msk = np.expand_dims(msk, axis=-1)

        # # - Check if there are enough cells in the segmentation
        # bin_msk = deepcopy(msk)
        # bin_msk[bin_msk > 0] = 1
        #
        # if bin_msk.sum() > MIN_CELL_PIXELS:
        imgs.append(img)
        msks.append(msk)

    data_tuples = np.array(list(zip(imgs, msks)), dtype=object)

    print(f'''
    =========
    = Stats =
    =========
    - Total data items: {len(data_tuples)}
    ''')

    return data_tuples


def clean_items_with_empty_masks(data_tuples: list, save_file: pathlib.Path = None):
    print('''
    ==========================================
    == Cleaning data items with empty masks ==
    ==========================================
    ''')
    data_tuples_pbar = tqdm(data_tuples)
    clean_imgs, clean_msks = list(), list()
    for img, msk in data_tuples_pbar:
        bin_msk = deepcopy(msk)
        bin_msk[bin_msk > 0] = 1

        # - Check if there are enough cells in the segmentation
        if bin_msk.sum() > MIN_CELL_PIXELS:
            clean_imgs.append(img)
            clean_msks.append(msk)

    clean_data_tuples = np.array(list(zip(clean_imgs, clean_msks)), dtype=object)
    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        np.save(str(save_file), clean_data_tuples)

    print(f'''
    =========
    = Stats =
    =========
    - {len(data_tuples) - len(clean_data_tuples)} items were filtered
    - Total clean data items: {len(clean_data_tuples)}
    ''')
    return clean_data_tuples


def get_runtime(seconds: float):
    hrs = int(seconds // 3600)
    mins = int((seconds - hrs * 3600) // 60)
    sec = seconds - hrs * 3600 - mins * 60

    # - Format the strings
    hrs_str = str(hrs)
    if hrs < 10:
        hrs_str = '0' + hrs_str
    min_str = str(mins)
    if mins < 10:
        min_str = '0' + min_str
    sec_str = f'{sec:.3}'
    if sec < 10:
        sec_str = '0' + sec_str

    return hrs_str + ':' + min_str + ':' + sec_str + '[H:M:S]'


def calc_seg_measure(gt_masks: np.ndarray, pred_masks: np.ndarray):
    """
    Converts a multi-class label into a one-hot labels for each object in the multi-class label
    :param: multi_class_mask - mask where integers represent different objects
    """
    # - Ensure the multi-class label is populated with int values
    gt_masks = gt_masks.astype(np.int16)
    pred_masks = pred_masks.astype(np.int16)

    # - Find the classes
    labels = np.unique(gt_masks)

    # - Discard the background (0)
    labels = labels[labels > 0]

    # - Convert the ground truth mask to one-hot class masks
    gt_one_hot_masks = []
    for lbl in labels:
        class_mask = deepcopy(gt_masks)
        class_mask[class_mask != lbl] = 0
        class_mask[class_mask > 0] = 1
        gt_one_hot_masks.append(class_mask)

    gt_one_hot_masks = np.array(gt_one_hot_masks, dtype=np.float32)
    if len(gt_one_hot_masks.shape) < 4:
        gt_one_hot_masks = np.expand_dims(gt_one_hot_masks, axis=0)

    # - Calculate the ground truth object area
    A = gt_one_hot_masks.sum(axis=(-3, -2, -1))
    A[A == 0] = np.nan

    # - Convert the predicted mask to one-hot class masks
    pred_one_hot_masks = []
    for lbl in labels:
        class_mask = deepcopy(pred_masks)
        class_mask[class_mask != lbl] = 0
        class_mask[class_mask > 0] = 1
        pred_one_hot_masks.append(class_mask)

    pred_one_hot_masks = np.array(pred_one_hot_masks, dtype=np.float32)
    if len(pred_one_hot_masks.shape) < 4:
        pred_one_hot_masks = np.expand_dims(pred_one_hot_masks, axis=0)

    # - Calculate the intersection
    I = np.logical_and(gt_one_hot_masks, pred_one_hot_masks).sum(axis=(-3, -2, -1))

    # - Calculate the union
    U = np.logical_or(gt_one_hot_masks, pred_one_hot_masks).sum(axis=(-3, -2, -1))

    # - Calculate the IoU
    U[U == 0] = 1  # to avoid division by 0
    IoU = I / U

    # - Fund invalid elements
    object_coverage_mask = I / A
    object_coverage_mask[object_coverage_mask <= 0.5] = 0
    object_coverage_mask[object_coverage_mask > 0.5] = 1

    # - Zero the invalid IoU
    IoU = object_coverage_mask * IoU

    seg_measure = np.nanmean(IoU, axis=0)
    if isinstance(seg_measure, np.ndarray):
        seg_measure[np.isnan(seg_measure)] = 0.0
    else:
        seg_measure = np.array([seg_measure])

    return seg_measure


def plot_seg_error(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, suptitle: str, title: str, figsize: tuple = (20, 20), save_file: pathlib.Path = None):
    # - Prepare the mask overlap image
    seg = np.zeros((*gt_mask.shape[:-1], 3))
    seg[..., 0] = gt_mask[..., 0]
    seg[..., 2] = pred_mask[..., 0]
    seg[seg > 0] = 1.

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.imshow(seg, alpha=0.3)
    ax.set(title=title)

    fig.suptitle(suptitle)

    if isinstance(save_file, pathlib.Path) and save_file.parent.is_dir():
        plt.savefig(save_file)
        plt.close()

    return fig, ax


def monitor_seg_error(gt_masks: np.ndarray, pred_masks: np.ndarray, seg_measures: np.ndarray, images: np.ndarray = None, n_samples: int = 5, figsize: tuple = (20, 10), save_dir: str or pathlib.Path = './seg_errors'):
    save_dir = pathlib.Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    data = list(zip(gt_masks, pred_masks, seg_measures))

    for idx, (gt, pred, seg_msr) in zip(np.arange(n_samples), data):
        seg = np.zeros((*gt.shape[:-1], 3))
        seg[..., 0] = gt[..., 0]
        seg[..., 2] = pred[..., 0]
        seg[seg > 0] = 1.

        if isinstance(images, np.ndarray):
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].imshow(images[idx], cmap='gray')
            ax[0].set(title='Original Image')
            ax[1].imshow(seg)
            ax[1].set(title=f'Seg Measure = {seg_msr:.4f}')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(seg)
            ax.set(title=f'Seg Measure = {seg_msr:.4f}')

        fig.suptitle(f'GT (red) vs Pred (blue) ')

        plt.savefig(save_dir / f'item_{idx}.png')
        plt.close()


def normalize(image: np.ndarray):
    return (image - image.mean() * np.max(image)) / (image.std() * np.max(image))


def calc_histogram(data: np.ndarray or list, bins: np.ndarray, normalize: bool = False):
    # - Plot histogram
    heights, ranges = np.histogram(data, range=(bins[0], bins[-1]), bins=len(bins), density=False)

    if normalize:
        heights = heights / len(data)

    return heights, ranges


def plot_hist(data: np.ndarray or list, bins: np.ndarray, save_file: pathlib.Path = None):
    # - Plot histogram
    ds = pd.DataFrame(dict(heights=data))

    rc = {
        'font.size': 12,
        'axes.labelsize': 20,
        'legend.fontsize': 20.,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    }
    sns.set_context(rc=rc)
    dist_plot = sns.displot(ds['heights'], bins=bins, rug=True, kde=True)

    if isinstance(save_file, pathlib.Path) and not save_file.is_file():
        os.makedirs(save_file.parent, exist_ok=True)
        dist_plot.savefig(save_file)
    else:
        print(f'WARNING: could not save plot to \'{save_file}\' as it already exists!')

    plt.close(dist_plot.figure)
    sns.set_context(rc=RC)


def to_numpy(data: np.ndarray, file_path: str or pathlib.Path, overwrite: bool = False, logger:  logging.Logger = None):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if isinstance(file_path, pathlib.Path):
        if overwrite or not file_path.is_file():
            os.makedirs(file_path.parent, exist_ok=True)
            np.save(str(file_path), data)
        else:
            warning_log(logger=logger, message=f'Could not save the data to the file \'{file_path}\' as it already exists (set overwrite=True to save on existing files in the future)! ')
    else:
        warning_log(logger=logger, message=f'Could not save the data to the file \'{file_path}\' - \'file_path\' must be in format str or pathlib.Path but is of type \'{type(file_path)}\'!')


def from_numpy(file_path: str or pathlib.Path, logger: logging.Logger = None):
    data = None

    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if isinstance(file_path, pathlib.Path):
        if file_path.is_file():
            data = np.load(str(file_path))
        else:
            warning_log(logger=logger, message=f'Could not load the data from \'{file_path}\' file - the file does not exists!')
    else:
        warning_log(logger=logger, message=f'Could not load the data from \'{file_path}\' file - the file must be of type str or pathlib.Path, but is of type \'{type(file_path)}\'!')

    return data


def to_pickle(file, name: str, save_dir: str or pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    pkl.dump(file, (save_dir / (name + '.pkl')).open(mode='wb'))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--gpu_id', type=int, default=0 if torch.cuda.device_count() > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')

    parser.add_argument('--load_checkpoint', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at the checkpoint file')
    parser.add_argument('--lunch_tb', default=False, action='store_true', help=f'If to lunch tensorboard')
    parser.add_argument('--train_data_dir', type=str, default=TRAIN_DATA_DIR, help='The path to the train data file')
    parser.add_argument('--test_data_dir', type=str, default=TEST_DATA_DIR, help='The path to the custom test file')
    parser.add_argument('--inference_data_dir', type=str, default=INFERENCE_DATA_DIR, help='The path to the inference data dir')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT, help='The height of the images that will be used for network training and inference. If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')
    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH, help='The width of the images that will be used for network training and inference. If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')

    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help='The number of channels in an input image (e.g., 3 for RGB, 1 for Grayscale etc)')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS, help='The number of channels in the output image (e.g., 3 for RGB, 1 for Grayscale etc)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=TRAIN_BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--tr_checkpoint_file', type=str, default=tr_configs.CHECKPOINT_FILE_BEST_MODEL, help=f'The path to the file which contains the checkpoints of the model')
    parser.add_argument('--tf_checkpoint_dir', type=str, default=tf_configs.CHECKPOINT_DIR, help=f'The path to the directory which contains the checkpoints of the model')
    parser.add_argument('--tf_checkpoint_file', type=str, default=tf_configs.CHECKPOINT_FILE_BEST_MODEL, help=f'The path to the file which contains the checkpoints of the model')

    # - DROP BLOCK
    parser.add_argument('--drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, default=DROP_BLOCK_KEEP_PROB, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, default=DROP_BLOCK_BLOCK_SIZE, help=f'The size of the block to drop')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'], default=OPTIMIZER, help=f'The optimizer to use')
    parser.add_argument('--weighted_loss', default=False, action='store_true', help=f'If to use the weighted version of the MSE loss')

    parser.add_argument('--optimizer_lr', type=float, default=OPTIMIZER_LR, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--optimizer_lr_decay', type=float, default=OPTIMIZER_LR_DECAY, help=f'The learning rate decay for Adagrad optimizer')
    parser.add_argument('--optimizer_beta_1', type=float, default=OPTIMIZER_BETA_1, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, default=OPTIMIZER_BETA_2, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_rho', type=float, default=OPTIMIZER_RHO, help=f'The decay rate (Adadelta, RMSprop)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')

    parser.add_argument('--optimizer_weight_decay', type=float, default=OPTIMIZER_WEIGHT_DECAY, help=f'The weight decay for ADAM, NADAM')
    parser.add_argument('--optimizer_momentum', type=float, default=OPTIMIZER_MOMENTUM, help=f'The momentum for SGD')
    parser.add_argument('--optimizer_dampening', type=float, default=OPTIMIZER_DAMPENING, help=f'The dampening for momentum')
    parser.add_argument('--optimizer_momentum_decay', type=float, default=OPTIMIZER_MOMENTUM_DECAY, help=f'The momentum for NADAM')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    # - KERNEL REGULARIZER
    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], default=KERNEL_REGULARIZER_TYPE, help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, default=KERNEL_REGULARIZER_L1, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, default=KERNEL_REGULARIZER_L2, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, default=KERNEL_REGULARIZER_FACTOR, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], default=KERNEL_REGULARIZER_MODE, help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    # - ACTIVATION
    parser.add_argument('--activation', type=str, choices=['swish', 'relu', 'leaky_relu'], default=ACTIVATION, help=f'The activation to use')
    parser.add_argument('--activation_relu_max_value', type=float, default=ACTIVATION_RELU_MAX_VALUE, help=f'The negative slope in the LeakyReLU activation function for values < 0')
    parser.add_argument('--activation_relu_negative_slope', type=float, default=ACTIVATION_RELU_NEGATIVE_SLOPE, help=f'The negative slope in the ReLU activation function for values < 0')
    parser.add_argument('--activation_relu_threshold', type=float, default=ACTIVATION_RELU_THRESHOLD, help=f'The value that has to be exceeded activate the neuron')
    parser.add_argument('--activation_leaky_relu_alpha', type=float, default=ACTIVATION_LEAKY_RELU_ALPHA, help=f'The negative slope in the LeakyReLU activation function for values < 0')

    return parser
