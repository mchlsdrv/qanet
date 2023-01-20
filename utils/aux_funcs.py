import datetime
import io
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
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import grey_erosion
from scipy.stats import pearsonr
from tqdm import tqdm

from global_configs.general_configs import (
    # OUTPUT_DIR,
    # CROP_WIDTH,
    # CROP_HEIGHT,
    # IN_CHANNELS,
    # OUT_CHANNELS,
    # EPOCHS,
    # TRAIN_BATCH_SIZE,
    # VAL_PROP,
    # OPTIMIZER_LR,
    # OPTIMIZER_LR_DECAY,
    # OPTIMIZER_BETA_1,
    # OPTIMIZER_BETA_2,
    # OPTIMIZER_RHO,
    # OPTIMIZER_WEIGHT_DECAY,
    # OPTIMIZER_MOMENTUM,
    # OPTIMIZER_DAMPENING,
    # OPTIMIZER_MOMENTUM_DECAY,
    # OPTIMIZER,
    # DROP_BLOCK_KEEP_PROB,
    # DROP_BLOCK_BLOCK_SIZE,
    # KERNEL_REGULARIZER_TYPE,
    # KERNEL_REGULARIZER_L1,
    # KERNEL_REGULARIZER_L2,
    # KERNEL_REGULARIZER_FACTOR,
    # KERNEL_REGULARIZER_MODE,
    # ACTIVATION,
    # ACTIVATION_RELU_MAX_VALUE,
    # ACTIVATION_RELU_NEGATIVE_SLOPE,
    # ACTIVATION_RELU_THRESHOLD,
    # ACTIVATION_LEAKY_RELU_ALPHA,
    # INFERENCE_DATA_DIR,
    MIN_CELL_PIXELS,
    # TRAIN_DATA_DIR,
    # TEST_DATA_DIR,
    # MASKS_DIR,
    HYPER_PARAMS_FILE,
)

# from pytorch.configs import general_configs as tr_configs
# from tensor_flow.configs import general_configs as tf_configs
import warnings

mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by the wrong thread" exception
plt.style.use('seaborn')  # <= using the seaborn plot style

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

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


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    # plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_figure_to_tensorboard(writer, figure, tag: str, step: int):
    with tf.device('/cpu:0'):
        with writer.as_default():
            # -> Write the scatter plot
            tf.summary.image(
                tag,
                get_image_from_figure(figure=figure),
                step=step
            )


def get_range(value, ranges: np.ndarray):
    for idx, (rng_min, rng_max) in enumerate(ranges):
        if rng_min < value < rng_max:
            break
    return rng_min, rng_max, idx


def str_2_float(str_val: str):
    return float(str_val.replace('_', '.'))


def float_2_str(float_val: float):
    return f'{float_val:.6f}'.replace('.', '_')


def check_unique_file(file: pathlib.Path or str):
    assert_pathable(argument=file, argument_name='file')
    fl = str_2_path(path=file)
    parent_dir = fl.parent

    idx = 0
    while fl.is_file():
        # - Create a file name
        f_name = get_file_name(path=file) + f'-{idx}.tif'

        # - Create the file to save the image
        fl = parent_dir / f_name

        idx += 1

    return fl


def check_iterable(data):
    return isinstance(data, np.ndarray) or isinstance(data, list)


def assert_iterable(argument, argument_name: str):
    assert check_iterable(data=argument), f'{argument_name} argument expected to be of type np.ndarray or list, but is of type \'{type({argument_name})}\'!'


def check_pathable(path):
    return isinstance(path, pathlib.Path) or isinstance(path, str)


def assert_pathable(argument, argument_name: str):
    assert check_pathable(path=argument), f'\'{argument_name}\' argument must be of type pathlib.Path or str, but is \'{type({argument})}\'!'


def get_parent_dir_name(path: pathlib.Path or str):
    assert_pathable(argument=path, argument_name='path')  # isinstance(path, pathlib.Path) or isinstance(path, str), f'\'path\' argument must be of type pathlib.Path or str, but is \'{type(path)}\''

    path = str_2_path(path=path)
    parent_dir = str(path.parent)
    parent_dir_name = parent_dir[::-1][:parent_dir[::-1].index('/')][::-1]
    return parent_dir_name


def get_file_name(path: pathlib.Path or str):
    assert_pathable(argument=path, argument_name='path')
    # assert isinstance(path, pathlib.Path) or isinstance(path, str), f'\'path\' argument must be of type pathlib.Path or str, but is \'{type(path)}\''

    path = str(path)
    file_name = path[::-1][path[::-1].index('.') + 1:path[::-1].index('/')][::-1]

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


def normalize_image(image: np.ndarray):
    return (image - image.mean()) / (image.std())


def image_clip_values(image: np.ndarray, max_val: int):
    image[image > max_val] = max_val
    return image


def image_2_float(image: np.ndarray, max_val: int = 255):
    return image_clip_values(image=image, max_val=max_val) / max_val


def load_image(image_file, add_channels: bool = False):
    img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)

    if add_channels and len(img.shape) < 3:
        img = add_channels_dim(image=img)
    return img


def get_split_data(data: np.ndarray or list, n_items: int):
    assert_iterable(argument=data, argument_name='data')

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=object)

    n_max = len(data)
    split_data = []
    for idx_min in range(0, n_max, n_items):
        split_data.append(data[idx_min:idx_min + n_items] if idx_min + n_items < n_max else data[idx_min:n_max])
    return split_data


def show_images(images, labels, suptitle='', figsize=(25, 10), save_file: pathlib.Path or str = None, verbose: bool = False, logger: logging.Logger = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=pathlib.Path(save_file), close_figure=True, verbose=verbose, logger=logger)


def line_plot(x: list or np.ndarray, ys: list or np.ndarray, suptitle: str, labels: list, colors: tuple = ('r', 'g', 'b'), save_file: pathlib.Path or str = None, logger: logging.Logger = None):
    # assert len(ys) == len(labels) == len(colors), f'The number of data items ({len(ys)}) does not match the number of labels ({len(labels)}) or the number of colors ({len(colors)})!'

    fig, ax = plt.subplots()
    for y, lbl, clr in zip(ys, labels, colors):
        ax.plot(x, y, color=clr, label=lbl)

    plt.legend()

    try:
        save_figure(figure=fig, save_file=save_file, close_figure=False, logger=logger)
    except Exception as err:
        err_log(logger=logger, message=f'{err}')


def get_hit_rate_plot_figure(true: np.ndarray, pred: np.ndarray, hit_rate_percent: int = None, figsize: tuple = (15, 15), logger: logging.Logger = None):
    # - Calculate the absolute error of true vs pred
    abs_err = np.abs(true - pred)

    # - Create a histogram of the absolute errors
    abs_err_hist, abs_err_tolerance = np.histogram(abs_err, bins=100, range=(0., 1.))

    # - Normalize the histogram
    abs_err_prob = abs_err_hist / np.sum(abs_err_hist)

    # - Plot the histogram
    fig, ax = plt.subplots(figsize=figsize)

    # > Calculate the cumulative probability of the density function of the absolute errors
    abs_err_cum_sum_prob = np.cumsum(abs_err_prob)
    ax.plot(abs_err_tolerance[:-1], abs_err_cum_sum_prob, linewidth=2)
    ax.set(xlabel='Absolute Error Tolerance', xlim=(0, 1), xticks=np.arange(0.0, 1.2, 0.2), ylabel='Hit Rate', ylim=(0, 1), yticks=np.arange(0.2, 1.2, 0.2))

    # - Add a line representing the hit rate percentage with corresponding AET value
    if isinstance(hit_rate_percent, int):
        # > Find the index of the hit_rate_percent
        abs_err_cum_sum_pct_idx = np.argwhere(abs_err_cum_sum_prob >= hit_rate_percent / 100).flatten().min()

        # > Find the real value of the hit_rate_percent
        cum_sum_err_pct = abs_err_cum_sum_prob[abs_err_cum_sum_pct_idx]

        # > Find the corresponding Absolute Error Tolerance to the hit_rate_percent value
        abs_err_tolerance_pct = abs_err_tolerance[abs_err_cum_sum_pct_idx]

        # > Plot the horizontal line for the hit rate percentage
        ax.axhline(cum_sum_err_pct, xmax=abs_err_tolerance_pct)

        # > Plot the vertical line for the corresponding AET value
        ax.axvline(abs_err_tolerance_pct, ymax=cum_sum_err_pct)

        # > Add the corresponding AET value
        ax.text(x=abs_err_tolerance_pct, y=cum_sum_err_pct, s=f'AET={abs_err_tolerance_pct:.3f}')

    return fig, abs_err_hist, abs_err


def get_scatter_plot_figure(x: np.ndarray, y: np.ndarray, plot_type: str, logger: logging.Logger = None):
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

    # - Calculate pearson correlation
    rho, p = pearsonr(x, y)

    # - Calculate mean squared error
    mse = np.mean(np.square(x[:10] - y[:10]))

    g.ax_joint.annotate(
        f'$\\rho = {rho:.3f}, MSE = {mse:.3f}$',
        xy=(0.1, 0.9),
        xycoords='axes fraction',
        ha='left',
        va='center',
        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
    )

    return g.figure, rho, p, mse


def get_files_under_dir(dir_path: pathlib.Path or str):
    fls = []
    for root, dirs, files in os.walk(str(dir_path), topdown=False):
        for fl in files:
            fls.append(fl)
    fls = np.array(fls, dtype=object)
    return fls


def str_2_path(path: str):
    assert_pathable(argument=path, argument_name='path')

    path = pathlib.Path(path) if isinstance(path, str) else path

    if path.is_dir():
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path.parent, exist_ok=True)

    return path


def save_figure(figure, save_file: pathlib.Path or str, overwrite: bool = False, close_figure: bool = False, verbose: bool = False, logger: logging.Logger = None):
    # - Convert save_file to path
    save_file = str_2_path(path=save_file)

    if isinstance(save_file, pathlib.Path):
        # - If the file does not exist or can be overwritten
        if not save_file.is_file() or overwrite:

            # - Create sub-path of the save file
            os.makedirs(save_file.parent, exist_ok=True)

            figure.savefig(str(save_file))
            if close_figure:
                plt.close(figure)
            if verbose:
                info_log(logger=logger, message=f'Figure was saved to \'{save_file}\'')
        elif verbose:
            info_log(logger=logger, message=f'Can not save figure - file \'{save_file}\' already exists and overwrite = {overwrite}!')
    elif verbose:
        info_log(logger=logger, message=f'Can not save figure - save_file argument must be of type pathlib.Path or str, but {type(save_file)} was provided!')


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

                        # - Configure the name of the segmentation files
                        if len(image_prefix) > 1:
                            seg_name = file.replace(image_prefix, seg_prefix)
                        else:
                            seg_name = seg_prefix + file[file.index(image_prefix)+1:]

                        # - If there was provided a sub-dir - add it
                        if seg_sub_dir is not None:
                            seg_fl = pathlib.Path(f'{seg_dir}/{seg_sub_dir}/{seg_name}')
                        else:
                            seg_fl = pathlib.Path(f'{seg_dir}/{seg_name}')

                        # - If there is a segmentation file with that path - add it
                        if img_fl.is_file() and seg_fl.is_file():
                            file_tuples.append((img_fl, seg_fl))
    return file_tuples


def get_data_dict(data_file_tuples: list):
    print('''
    ====================================================
    == Loading images and corresponding segmentations ==
    ====================================================
    ''')
    data_dict = dict()
    files_pbar = tqdm(data_file_tuples)
    for img_fl, msk_fl in files_pbar:
        img = load_image(image_file=img_fl, add_channels=True)

        msk = load_image(image_file=msk_fl, add_channels=True)

        data_dict[str(img_fl)] = (img, str(msk_fl), msk)

    print(f'''
    =========
    = Stats =
    =========
    - Total data items: {len(data_dict)}
    ''')

    return data_dict


def get_relevant_data(data_dict: dict, relevant_files: list, save_file: pathlib.Path = None, logger: logging.Logger = None):
    print('''
    ==================================
    == Cleaning non-existing images ==
    ==================================
    ''')
    data_tuples_pbar = tqdm(data_dict)
    relevant_data_dict = dict()
    for img_fl_key in data_tuples_pbar:
        # - Unpack the data
        img_fl = pathlib.Path(img_fl_key)
        img_parent_dir = get_parent_dir_name(path=img_fl)
        img_name = get_file_name(path=img_fl)

        if f'{img_parent_dir}_{img_name}' in relevant_files:
            relevant_data_dict[img_fl_key] = data_dict.get(img_fl_key)

    to_pickle(data=relevant_data_dict, save_file=save_file, logger=logger)

    print(f'''
    =========
    = Stats =
    =========
    - {len(data_dict) - len(relevant_data_dict)} items were filtered
    - Total clean data items: {len(relevant_data_dict)}
    ''')
    return relevant_data_dict


def clean_items_with_empty_masks(data_dict: dict, save_file: pathlib.Path = None, logger: logging.Logger = None):
    print('''
    ==========================================
    == Cleaning data items with empty masks ==
    ==========================================
    ''')
    data_tuples_pbar = tqdm(data_dict)
    clean_data_dict = dict()
    for img_fl_key in data_tuples_pbar:
        # - Unpack the data
        img, msk_fl, msk = data_dict.get(img_fl_key)

        # - Construct the binary mask
        bin_msk = deepcopy(msk)
        bin_msk[bin_msk > 0] = 1

        # - Check if there are enough cells in the segmentation
        if bin_msk.sum() > MIN_CELL_PIXELS:
            clean_data_dict[str(img_fl_key)] = (img, str(msk_fl), msk)

    to_pickle(data=clean_data_dict, save_file=save_file, logger=logger)

    print(f'''
    =========
    = Stats =
    =========
    - {len(data_dict) - len(clean_data_dict)} items were filtered
    - Total clean data items: {len(clean_data_dict)}
    ''')
    return clean_data_dict


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


def merge_categorical_masks(masks: np.ndarray):
    msk = np.zeros(masks.shape[1:], dtype=np.float32)
    for cat_mks in masks:
        msk += cat_mks

    # - Fix the boundary if it was added several times from different cells
    msk[msk > 2.] = 2.
    msk = msk.astype(np.float32)

    return msk


def split_instance_mask(instance_mask: np.ndarray, labels: np.ndarray or list = None):
    """
    Splits an instance mask into N binary masks
    :param: instance_mask - mask where integers represent different objects
    """
    # - Ensure the multi-class label is populated with int values
    inst_msk = instance_mask.astype(np.int16)

    # - Find the classes
    lbls = labels
    if not check_iterable(data=labels):
        lbls = np.unique(inst_msk)

    # - Discard the background (0)
    lbls = lbls[lbls > 0]

    # - Convert the ground truth mask to one-hot class masks
    bin_masks = [np.zeros_like(inst_msk)]
    for lbl in lbls:
        bin_class_mask = deepcopy(inst_msk)
        bin_class_mask[bin_class_mask != lbl] = 0
        bin_class_mask[bin_class_mask > 0] = 1
        bin_masks.append(bin_class_mask)

    bin_masks = np.array(bin_masks)

    return bin_masks


def get_categorical_mask(binary_mask: np.ndarray):
    # Shrinks the labels
    inner_msk = grey_erosion(binary_mask, size=2)

    # Create the contur of the cells
    contur_msk = binary_mask - inner_msk
    contur_msk[contur_msk > 0] = 2

    # - Create the inner part of the cell
    inner_msk[inner_msk > 0] = 1

    # - Combine the inner and the contur masks to create the categorical mask with three classes, i.e., background 0, inner 1 and contur 2
    cat_msk = inner_msk + contur_msk

    return cat_msk


def enhance_contrast(image: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(image)

    return img


def instance_2_categorical(masks: np.ndarray or list):
    """
    Converts an instance masks (i.e., where each cell is represented by a different color, to a mask with 3 classes, i.e.,
        - 0 for background,
        - 1 for the inner part of the cell
        - 2 for cells' boundary
    """
    btch_cat_msks = []
    for msk in masks:
        # - Split the mask into binary masks of separate cells
        bin_msks = split_instance_mask(instance_mask=msk)

        # - For each binary cell - get a categorical mask
        cat_msks = [np.zeros_like(msk)]
        for bin_msk in bin_msks:
            cat_msk = get_categorical_mask(binary_mask=bin_msk)
            cat_msks.append(cat_msk)

        # - Merge the categorical masks for each cell into a single categorical mask
        mrgd_cat_msk = merge_categorical_masks(masks=np.array(cat_msks))

        # - Add the categorical mask to the batch masks
        btch_cat_msks.append(mrgd_cat_msk)

    # - Convert to array
    btch_cat_msks = np.array(btch_cat_msks, dtype=np.float32)

    return btch_cat_msks


def calc_seg_score(gt_masks: np.ndarray, pred_masks: np.ndarray):
    """
    Converts a multi-class label into a one-hot labels for each object in the multi-class label
    :param: multi_class_mask - mask where integers represent different objects
    """
    # - Ensure the multi-class label is populated with int values
    gt_masks = gt_masks.astype(np.int16)
    pred_masks = pred_masks.astype(np.int16)

    # - Find the classes
    lbls = np.unique(gt_masks)

    # - Discard the background (0)
    lbls = lbls[lbls > 0]

    # - Convert the ground truth mask to one-hot class masks
    gt_one_hot_masks = split_instance_mask(instance_mask=gt_masks, labels=lbls)
    gt_one_hot_masks = np.array(gt_one_hot_masks, dtype=np.float32)
    if len(gt_one_hot_masks.shape) < 4:
        gt_one_hot_masks = np.expand_dims(gt_one_hot_masks, axis=0)

    # - Calculate the ground truth object area
    A = gt_one_hot_masks.sum(axis=(-3, -2, -1))

    # - Convert the predicted mask to one-hot class masks
    pred_one_hot_masks = split_instance_mask(instance_mask=pred_masks, labels=lbls)
    pred_one_hot_masks = np.array(pred_one_hot_masks, dtype=np.float32)
    if len(pred_one_hot_masks.shape) < 4:
        pred_one_hot_masks = np.expand_dims(pred_one_hot_masks, axis=0)

    # - Calculate an intersection for each object with each other object
    Is = (gt_one_hot_masks[:, np.newaxis, ...] * pred_one_hot_masks[np.newaxis, ...]).sum(axis=(-1, -2, -3))

    # - Calculate a union for each object with each other object
    Us = np.logical_or(gt_one_hot_masks[:, np.newaxis, ...], pred_one_hot_masks[np.newaxis, ...]).sum(axis=(-1, -2, -3))
    Us[Us == 0] = 1  # to avoid division by 0

    # - Calculate the IoU
    IoU = (Is / Us).max(axis=1)

    # - Find objects which more than a half of them is covered by a mask
    object_coverage_mask = Is.max(axis=1) / A
    valid_idxs = np.argwhere(object_coverage_mask > 0.5).flatten()

    # - Choose only the valid IoU
    IoU = IoU[valid_idxs]

    # - Among the valid IoUs - zero the once which are lower than a 0.5
    IoU[IoU <= 0.5] = 0

    # - Calculate the mean seg score
    seg_scr = np.nanmean(IoU, axis=0)
    if isinstance(seg_scr, np.ndarray):
        # - Replace np.nan with 0.
        seg_scr[np.isnan(seg_scr)] = 0.0
    else:
        seg_scr = np.array([seg_scr])

    return seg_scr


def update_hyper_parameters(hyper_parameters: dict, arguments: argparse.Namespace):
    # - Get hyper-parameter names
    hyp_param_categories = list(hyper_parameters.keys())

    # - Get the argument names
    args = vars(arguments)
    arg_names = list(args.keys())

    # - For each argument
    for arg_name in arg_names:
        for hyp_param_cat in hyp_param_categories:
            # - Get the hyperparameter names fo the category
            hyp_param_names = hyper_parameters.get(hyp_param_cat)

            # - If the argument name is in hyperparameter names for the current category
            if arg_name in hyp_param_names and args.get(arg_name) is not None:
                # - Update it with the relevant value
                hyper_parameters.get(hyp_param_cat)[arg_name] = args.get(arg_name)


def get_image_mask_figure(image: np.ndarray, mask: np.ndarray, suptitle: str = '', title: str = '', figsize: tuple = (20, 20)):
    # - Prepare the mask overlap image
    msk = np.zeros((*mask.shape[:-1], 3))

    # - Green channel - inner cell
    inner_msk = deepcopy(mask)
    inner_msk[inner_msk != 1] = 0
    msk[..., 1] = inner_msk[..., 0]

    # - Blue channel - contur of the cell
    contur_msk = deepcopy(mask)
    contur_msk[contur_msk != 2] = 0
    msk[..., 2] = contur_msk[..., 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.imshow(msk, alpha=0.3)
    ax.set(title=title)

    fig.suptitle(suptitle)

    return fig

    # save_figure(figure=fig, save_file=save_file)
    #
    # if isinstance(tensorboard_params, dict):
    #     write_figure_to_tensorboard(
    #         writer=tensorboard_params.get('writer'),
    #         figure=fig,
    #         tag=tensorboard_params.get('tag'),
    #         step=tensorboard_params.get('step')
    #     )
    #
    # plt.close(fig)


def plot_mask_error(image: np.ndarray, mask: np.ndarray, pred_mask: np.ndarray = None, suptitle: str = '', title: str = '', figsize: tuple = (20, 20), tensorboard_params: dict = None, save_file: pathlib.Path = None, overwrite: bool = False):
    # - Prepare the mask overlap image
    msk = np.zeros((*mask.shape[:-1], 3))
    msk[..., 2] = mask[..., 0]

    # - If there is a predicted segmentation
    if isinstance(pred_mask, np.ndarray):
        msk[..., 0] = pred_mask[..., 0]

    # - Convert instance segmentation to binary
    msk[msk > 0] = 1.

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.imshow(msk, alpha=0.3)
    ax.set(title=title)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=save_file)

    if isinstance(tensorboard_params, dict):
        write_figure_to_tensorboard(
            writer=tensorboard_params.get('writer'),
            figure=fig,
            tag=tensorboard_params.get('tag'),
            step=tensorboard_params.get('step')
        )

    plt.close(fig)


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


def plot_hist(data: np.ndarray or list, bins: np.ndarray, save_file: pathlib.Path = None, overwrite: bool = False):
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

    if isinstance(save_file, pathlib.Path) and (not save_file.is_file() or overwrite):
        os.makedirs(save_file.parent, exist_ok=True)
        dist_plot.savefig(save_file)
    else:
        print(f'WARNING: could not save plot to \'{save_file}\' as it already exists!')

    plt.close(dist_plot.figure)
    sns.set_context(rc=RC)


def to_numpy(data: np.ndarray, file_path: str or pathlib.Path, overwrite: bool = False, logger: logging.Logger = None):
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


def from_pickle(data_file: pathlib.Path or str, logger: logging.Logger = None):
    data = None
    if check_pathable(path=data_file):
        data_file = str_2_path(path=data_file)

        data = pkl.load(data_file.open(mode='rb'))
    else:
        warning_log(logger=logger, message=f'Could not load the data from file \'{data_file}\'!')

    return data


def to_pickle(data, save_file: str or pathlib.Path, logger: logging.Logger = None):
    if check_pathable(path=save_file):
        save_file = str_2_path(path=save_file)

        pkl.dump(data, save_file.open(mode='wb'))
    else:
        warning_log(logger=logger, message=f'Could not save the data to file \'{save_file}\'!')


def print_pretty_message(message: str, delimiter_symbol: str = '='):
    delimiter_len = len(message) + 6

    for i in range(delimiter_len):
        print(delimiter_symbol, end='')

    print(f'\n{delimiter_symbol}{delimiter_symbol} {message} {delimiter_symbol}{delimiter_symbol}')

    for i in range(delimiter_len):
        print(delimiter_symbol, end='')

    print('')


def get_data(mode: str, hyper_parameters: dict, logger: logging.Logger = None):
    data_dict = dict()

    dt_fl = str_2_path(path=hyper_parameters.get(mode)['temp_data_file'])
    if not hyper_parameters.get('data')['reload_data'] and dt_fl.is_file():
        data_dict = from_pickle(data_file=dt_fl, logger=logger)
    else:
        dt_dir = str_2_path(path=hyper_parameters.get(mode)['data_dir'])
        fl_tupls = scan_files(
            root_dir=dt_dir,
            seg_dir_postfix=hyper_parameters.get(mode)['seg_dir_postfix'],
            image_prefix=hyper_parameters.get(mode)['image_prefix'],
            seg_prefix=hyper_parameters.get(mode)['seg_prefix'],
            seg_sub_dir=hyper_parameters.get(mode)['seg_sub_dir']
        )

        np.random.shuffle(fl_tupls)

        # - Load images and their masks
        data_dict = get_data_dict(data_file_tuples=fl_tupls)

        if mode == 'training':
            # - Clean data items with no objects in them
            data_dict = clean_items_with_empty_masks(data_dict=data_dict, save_file=hyper_parameters.get(mode)['temp_data_file'])

            data_dict = get_relevant_data(data_dict=data_dict, relevant_files=os.listdir(hyper_parameters.get(mode)['mask_dir']), save_file=hyper_parameters.get(mode)['temp_data_file'], logger=logger)

    return data_dict


def categorical_2_rgb(mask: np.ndarray):
    rgb_msk = np.zeros((*mask.shape[:2], 3))

    # - Background - red channel
    msk_bg = deepcopy(mask)
    msk_bg += 1
    msk_bg[msk_bg > 1] = 0
    rgb_msk[..., 0] = msk_bg

    # - Inner part of the cell - blue channel
    msk_in = deepcopy(mask)
    msk_in[msk_in != 1] = 0
    rgb_msk[..., 1] = msk_in

    # - Outer part of the cell - blue channel
    msk_out = deepcopy(mask)
    msk_out[msk_out != 2] = 0
    rgb_msk[..., 2] = msk_out

    return rgb_msk


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--name', type=str, help='The name of the experiment')
    parser.add_argument('--debug', default=False, action='store_true', help=f'If the run is a debugging run')
    parser.add_argument('--gpu_id', type=int, default=0 if torch.cuda.device_count() > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--hyper_params_file', type=str, default=HYPER_PARAMS_FILE, help='The path to the file with the hyper-parameters')
    parser.add_argument('--reload_data', default=False, action='store_true', help=f'If to reload the data')

    parser.add_argument('--in_train_augmentation', default=False, action='store_true', help=f'Regular mode where the augmentation is performed on-the-fly')
    parser.add_argument('--wandb', default=False, action='store_true', help=f'If to use the wandb callback')
    parser.add_argument('--tensorboard', default=False, action='store_true', help=f'If to use the tensorboard callback')
    parser.add_argument('--load_checkpoint', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at the checkpoint file')
    parser.add_argument('--train_data_dir', type=str, help='The path to the train data file')
    parser.add_argument('--train_image_dir', type=str, help='The path to the train images directory')
    parser.add_argument('--train_mask_dir', type=str, help='The path to the train masks directory')
    parser.add_argument('--train_temp_data_file', type=str, help='The path to the train data file')
    parser.add_argument('--test_data_dir', type=str, help='The path to the custom test file')
    parser.add_argument('--inference_data_dir', type=str, help='The path to the inference data dir')

    parser.add_argument('--output_dir', type=str, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--crop_height', type=int, help='The height of the images that will be used for network training and inference. If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')
    parser.add_argument('--crop_width', type=int, help='The width of the images that will be used for network training and inference. If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')

    parser.add_argument('--in_channels', type=int, help='The number of channels in an input image (e.g., 3 for RGB, 1 for Grayscale etc)')
    parser.add_argument('--out_channels', type=int, help='The number of channels in the output image (e.g., 3 for RGB, 1 for Grayscale etc)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, help='The number of samples in each batch')
    parser.add_argument('--val_prop', type=float, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--tr_checkpoint_file', type=str, help=f'The path to the file which contains the checkpoints of the model')
    parser.add_argument('--tf_checkpoint_dir', type=str, help=f'The path to the directory which contains the checkpoints of the model')
    parser.add_argument('--tf_checkpoint_file', type=str, help=f'The path to the file which contains the checkpoints of the model')

    # - DROP BLOCK
    parser.add_argument('--drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, help=f'The size of the block to drop')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'], help=f'The optimizer to use')
    parser.add_argument('--weighted_loss', default=False, action='store_true', help=f'If to use the weighted version of the MSE loss')

    parser.add_argument('--optimizer_lr', type=float, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--optimizer_lr_decay', type=float, help=f'The learning rate decay for Adagrad optimizer')
    parser.add_argument('--optimizer_beta_1', type=float, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_rho', type=float, help=f'The decay rate (Adadelta, RMSprop)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')

    parser.add_argument('--optimizer_weight_decay', type=float, help=f'The weight decay for ADAM, NADAM')
    parser.add_argument('--optimizer_momentum', type=float, help=f'The momentum for SGD')
    parser.add_argument('--optimizer_dampening', type=float, help=f'The dampening for momentum')
    parser.add_argument('--optimizer_momentum_decay', type=float, help=f'The momentum for NADAM')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    # - KERNEL REGULARIZER
    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    # - ACTIVATION
    parser.add_argument('--activation', type=str, choices=['swish', 'relu', 'leaky_relu'], help=f'The activation to use')
    parser.add_argument('--activation_relu_max_value', type=float, help=f'The negative slope in the LeakyReLU activation function for values < 0')
    parser.add_argument('--activation_relu_negative_slope', type=float, help=f'The negative slope in the ReLU activation function for values < 0')
    parser.add_argument('--activation_relu_threshold', type=float, help=f'The value that has to be exceeded activate the neuron')
    parser.add_argument('--activation_leaky_relu_alpha', type=float, help=f'The negative slope in the LeakyReLU activation function for values < 0')

    return parser
