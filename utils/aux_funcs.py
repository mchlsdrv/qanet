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
import cv2
import yaml

import matplotlib.pyplot as plt
from scipy.ndimage import grey_erosion
from scipy.stats import pearsonr
from tqdm import tqdm

from configs.general_configs import (
    MIN_CELL_PIXELS,
    HYPER_PARAMS_FILE,
    EPSILON,
    COLUMN_NAMES,
)

import warnings

import logging
import logging.config
from functools import partial

import torch

from configs.general_configs import OPTIMIZER_EPS
from utils.model_utils import LitRibCage

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

MIN_OBJ_PCT = 1
DEBUG_AUX_FUNCS = False
# DEBUG_AUX_FUNCS = True


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_device(gpu_id: int = 0, logger: logging.Logger = None):
    n_gpus = torch.cuda.device_count()

    print('> Available GPUs:')
    print(f'\t- Number of GPUs: {n_gpus}\n')
    device = 'cpu'
    if n_gpus > 0:
        try:
            if -1 < gpu_id < n_gpus - 1:
                print(f'> Setting GPU to: {gpu_id}\n')

                device = f'cuda:{gpu_id}'

                print(f'''
    ========================
    == Running on {device}  ==
    ========================
                ''')
            elif gpu_id < 0:
                device = 'cpu'
                print(f'''
    ====================
    == Running on CPU ==
    ====================
                        ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)

    return device


def get_optimizer(algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'sgd':
        optimizer = partial(
            torch.optim.SGD,
            lr=args.get('lr'),
            momentum=args.get('momentum'),
            weight_decay=args.get('weight_decay'),
            nesterov=args.get('nesterov'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adam':
        optimizer = partial(
            torch.optim.Adam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamw':
        optimizer = partial(
            torch.optim.AdamW,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'sparse_adam':
        optimizer = partial(
            torch.optim.SparseAdam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            torch.optim.NAdam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            momentum_decay=args.get('momentum_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            torch.optim.Adamax,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adadelta':
        optimizer = partial(
            torch.optim.Adadelta,
            lr=args.get('lr'),
            rho=args.get('rho'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adagrad':
        optimizer = partial(
            torch.optim.Adadelta,
            lr=args.get('lr'),
            lr_decay=args.get('lr_decay'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    return optimizer


def get_range(value, ranges: np.ndarray):
    rng_min, rng_max, idx = 0, 0, 0
    for idx, (rng_min, rng_max) in enumerate(ranges):
        if rng_min < value < rng_max:
            break
    return rng_min, rng_max, idx


def str_2_float(str_val: str):
    return float(str_val.replace('-', '').replace('_', '.'))


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
    assert check_iterable(data=argument), \
        f'{argument_name} argument expected to be of type np.ndarray or list,' \
        f' but is of type \'{type({argument_name})}\'!'


def check_pathable(path):
    return isinstance(path, pathlib.Path) or isinstance(path, str)


def assert_pathable(argument, argument_name: str):
    assert check_pathable(path=argument), f'\'{argument_name}\' argument must' \
                                          f' be of type pathlib.Path or str,' \
                                          f' but is \'{type({argument})}\'!'


def get_parent_dir_name(path: pathlib.Path or str):
    assert_pathable(argument=path, argument_name='path')

    path = str_2_path(path=path)
    parent_dir = str(path.parent)
    parent_dir_name = parent_dir[::-1][:parent_dir[::-1].index('/')][::-1]
    return parent_dir_name


def get_file_name(path: pathlib.Path or str):
    # - If the file does not represent a path - through an error
    assert_pathable(argument=path, argument_name='path')

    #  - Convert to the str object
    path = str(path)

    # - Assign to the return value
    file_name = path

    # - Discard the path if present
    if '/' in path:
        file_name = path[::-1][:path[::-1].index('/')][::-1]

    # - Discard the file extension if present
    if '.' in path:
        file_name = file_name[::-1][file_name[::-1].index('.') + 1:][::-1]

    return file_name


def get_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_channels_dim(image: np.ndarray):
    # - If the image is 2D - add the channel dimension
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=-1)
    return image


def standardize_image(image: np.ndarray):
    return (image - image.mean()) / (image.std() + EPSILON)


# noinspection PyArgumentList
def min_max_normalize_image(image: np.ndarray):
    return (image - image.min()) / (image.max() - image.min() + EPSILON)


def image_clip_values(image: np.ndarray, max_val: int):
    image[image > max_val] = max_val
    return image


def image_2_float(image: np.ndarray, max_val: int = 255):
    return image / max_val if max_val != 0 else image


def adjust_brightness_(image, delta):
    out_img = image + delta
    return out_img


def adjust_contrast_(image, factor):
    img_mean = image.mean()
    out_img = (image - img_mean) * factor + img_mean
    return out_img


def transform_image(image: np.ndarray, augment: bool = False):
    # - Make sure the image is in the right range
    # img = image_clip_values(image=image, max_val=255)

    # - Convert the image to float dividing by it by 255
    img = image.astype(np.float32)

    # - Standardize the image by (I - E[I]) / std(I)
    # img = standardize_image(image=img)

    if augment:
        # - Random contrast plus/minus 50%
        random_contrast_factor = np.random.rand() + 0.5
        img = adjust_contrast_(img, random_contrast_factor)

        # - Random brightness delta plus/minus 10% of maximum value
        random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img.max()
        img = adjust_brightness_(img, random_brightness_delta)
    return img


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
        split_data.append(
            data[idx_min:idx_min + n_items] if idx_min + n_items < n_max
            else data[idx_min:n_max])
    return split_data


def print_results(results: pd.DataFrame, rho: float, p: float, mse: float):
    true_seg_scr_mu, pred_seg_scr_mu = results.loc[:, "seg_score"].mean(), \
        results.loc[:, "pred_seg_score"].mean()
    true_seg_scr_std, pred_seg_scr_std = results.loc[:, "seg_score"].std(), \
        results.loc[:, "pred_seg_score"].std()
    print(f'''
    ***********************************************************************
    ******************************* RESULTS *******************************
    ***********************************************************************
    ** > Mean Seg Score Error (%): {100 - (true_seg_scr_mu * 100) /
                                    (pred_seg_scr_mu + EPSILON):.1f}%
    **    - True = {true_seg_scr_mu:.3f}±{true_seg_scr_std:.4f}
    **    - Predicted = {pred_seg_scr_mu:.3f}±{pred_seg_scr_std:.4f}
    ** > Pearson\'s correlation: 
    **    - rho  = {rho:.3f}
    **    - p  = {p:.3f}
    ** > MSE: {mse:.3f}
    ***********************************************************************
    ***********************************************************************
    ''')


def clear_unnecessary_columns(dataframe: pd.DataFrame):
    columns_2_drop = list()
    for clm_name in dataframe.columns.values:
        if clm_name not in COLUMN_NAMES:
            columns_2_drop.append(clm_name)

    dataframe = dataframe.drop(columns=columns_2_drop)
    return dataframe


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


def get_train_val_split(data_list: list or np.ndarray, validation_proportion: float = .2,
                        logger: logging.Logger = None):
    n_items = len(data_list)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * validation_proportion)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data_list, dtype=object)

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    # - The items for training are the once which are not included in the
    # validation set
    train_data = np_data[np.setdiff1d(item_idxs, val_idxs)]

    info_log(logger=logger,
             message=f'| Number of train data files : {len(train_data)} '
                     f'| Number of validation data files : {len(val_data)} |')

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


def build_metadata(data_dir: str or pathlib.Path, shape: tuple,
                   min_val: int or float = 0.0, max_val: int or float = 255.0):
    """
    Build a metadata file for a multiple directory data format.
    Expects the files in the following format under the data_dir:
        - Image files for each class in a directory 0x
        - Segmentations for each image in a directory 0x_XXXX_XXXX
    *  x - should be an integer
    ** X - may be anything
    """
    file_dict = {}
    for root, folders, _ in os.walk(data_dir):
        for fldr in folders:
            for sub_root, _, files in os.walk(f'{root}/{fldr}'):
                for file in files:
                    # dir_name = pathlib.Path(root).name
                    if file_dict.get(fldr[:2]) is None:
                        file_dict[fldr[:2]] = [f'{fldr}/{file}']
                    else:
                        file_dict.get(fldr[:2]).append(f'{fldr}/{file}')

    for dir_name in file_dict:
        # - Get the files in the current dir
        files = file_dict[dir_name]

        # - Sort them by image - segmentation
        files = sorted(sorted(files, key=lambda x: x[-7:]), key=lambda x: x[:2])

        # - Group the files as tuples of the format (image, segmentation)
        files = [[files[i], files[i + 1]] for i in range(0, len(files) - 1, 2)]

        # - Ensure that the image file will be placed before the corresponding
        # mask file
        files = [sorted(fl_tpl, key=lambda x: len(x)) for fl_tpl in files]

        # - Save the metadata file
        pkl.dump(dict(filelist=files, max=max_val, min=min_val, shape=shape),
                 pathlib.Path(
                     f'{data_dir}/metadata_{dir_name}.pkl').open(mode='wb'))


def get_files_from_metadata(root_dir: str or pathlib.Path, metadata_files_regex):
    img_seg_fls = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if re.match(metadata_files_regex, file) is not None:
                metadata_file = f'{root}/{file}'
                with pathlib.Path(metadata_file).open(mode='rb') as pkl_in:
                    metadata = pkl.load(pkl_in)
                    for metadata_tuple in metadata.get('filelist'):
                        img_seg_fls.append((f'{root}/{metadata_tuple[0]}',
                                            f'{root}/{metadata_tuple[1]}'))

    return img_seg_fls


def get_data_files(data_dir: str, metadata_configs: dict, val_prop: float = .2, logger: logging.Logger = None):
    build_metadata(data_dir=data_dir, shape=metadata_configs.get('shape'),
                   min_val=metadata_configs.get('min_val'),
                   max_val=metadata_configs.get('max_val'))

    train_fls, val_fls = get_train_val_split(
        data_list=get_files_from_metadata(
            root_dir=data_dir,
            metadata_files_regex=metadata_configs.get('regex'),
        ),
        val_prop=val_prop,
        logger=logger
    )
    return train_fls, val_fls


def get_model(hyper_parameters: dict):
    model = LitRibCage(
        in_channels=hyper_parameters.get('model')['in_channels'],
        out_channels=hyper_parameters.get('model')['out_channels'],
        input_image_shape=(hyper_parameters.get('data')['image_height'], hyper_parameters.get('data')['image_width']),
        conv2d_out_channels=hyper_parameters.get('model')['architecture']['conv2d_blocks']['out_channels'],
        conv2d_kernel_sizes=hyper_parameters.get('model')['architecture']['conv2d_blocks']['kernel_sizes'],
        fc_out_features=hyper_parameters.get('model')['architecture']['fc_blocks']['out_features'],
        optimizer=get_optimizer(
            algorithm=hyper_parameters.get('training')['optimizer'],
            args=dict(
                lr=hyper_parameters.get('training')['optimizer_lr'],
                lr_decay=hyper_parameters.get('training')['optimizer_lr_decay'],
                betas=(
                    hyper_parameters.get('training')['optimizer_beta_1'],
                    hyper_parameters.get('training')['optimizer_beta_2']),
                weight_decay=hyper_parameters.get('training')['optimizer_weight_decay'],
                momentum=hyper_parameters.get('training')['optimizer_momentum'],
                momentum_decay=hyper_parameters.get('training')['optimizer_momentum_decay'],
                dampening=hyper_parameters.get('training')['optimizer_dampening'],
                rho=hyper_parameters.get('training')['optimizer_rho'],
                nesterov=hyper_parameters.get('training')['optimizer_nesterov'],
                amsgrad=hyper_parameters.get('training')['optimizer_amsgrad']
            )
        ),
        output_dir=hyper_parameters.get('general')['output_dir'],
    )

    return model


def get_model_configs(configs_file: pathlib.Path, logger: logging.Logger):
    model_configs = None
    if configs_file.is_file():
        model_configs = read_yaml(configs_file)
        if model_configs is not None:
            info_log(logger=logger,
                     message=f'The model configs were successfully loaded from '
                             f'\'{configs_file}\'!')
        else:
            info_log(logger=logger,
                     message=f'The model configs file ({configs_file}) does not'
                             f' contain valid model configurations!')
    else:
        info_log(logger=logger,
                 message=f'The model configs file ({configs_file}) does not '
                         f'exist!')
    return model_configs


def get_image_mask_file_tuples(root_dir: pathlib.Path or str, seg_dir_postfix: str,
                               image_prefix: str, seg_prefix: str, seg_sub_dir: str = None):
    file_tuples = list()
    for root, folders, _ in os.walk(root_dir):
        root = pathlib.Path(root)
        for fldr in folders:
            seg_dir = root / f'{fldr}_{seg_dir_postfix}'
            if seg_dir.is_dir():
                for sub_root, _, files in os.walk(root / fldr):
                    for file in files:
                        img_fl = pathlib.Path(f'{sub_root}/{file}')

                        # - Configure the name of the segmentation files
                        if len(image_prefix) > 1:
                            seg_name = file.replace(image_prefix, seg_prefix)
                        else:
                            seg_name = seg_prefix + file[file.index(
                                image_prefix) + 1:]

                        # - If there was provided a sub-dir - add it
                        if seg_sub_dir is not None:
                            seg_fl = pathlib.Path(
                                f'{seg_dir}/{seg_sub_dir}/{seg_name}')
                        else:
                            seg_fl = pathlib.Path(f'{seg_dir}/{seg_name}')

                        # - If there is a segmentation file with that path -
                        # add it
                        if img_fl.is_file() and seg_fl.is_file():
                            file_tuples.append((img_fl, seg_fl))
    return file_tuples


def rename_string_column(dataframe: pd.DataFrame, column_name: str, old_str: str, new_str: str,
                         save_file: pathlib.Path or str = None):
    dataframe.loc[:, column_name] = dataframe.apply(
        lambda x: x[column_name].replace(old_str, new_str), axis=1)

    if check_pathable(save_file):
        dataframe.to_csv(save_file, index=False)


def get_data_dict(image_mask_file_tuples: list):
    print('\t\t- Loading images and corresponding segmentations')

    data_dict = dict()
    files_pbar = tqdm(image_mask_file_tuples)
    for img_fl, msk_fl in files_pbar:
        img = load_image(image_file=img_fl, add_channels=True)

        msk = load_image(image_file=msk_fl, add_channels=True)

        data_dict[str(img_fl)] = (img, str(msk_fl), msk)

    print(f'\t\t\t* Total data items: {len(data_dict)}')

    return data_dict


def get_relevant_data(data_dict: dict, relevant_files: list, save_file: pathlib.Path = None,
                      logger: logging.Logger = None):
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


def pad_image(image: np.ndarray, shape: tuple, pad_value: int = 0):
    h, w = image.shape
    img_padded = np.zeros(shape) * pad_value
    img_padded[:h, :w] = image

    return img_padded


def get_objects(image: np.ndarray, mask: np.ndarray, crop_height: int, crop_width: int,
                output_dir: pathlib.Path = None):
    # - Find all the objects in the mask
    (_, msk_cntd, stats, centroids) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), cv2.CV_16U)

    # - Convert centroids to int16 to be able to represent pixel locations
    cntrs = centroids.astype(np.int16)
    if DEBUG_AUX_FUNCS:
        print(f'cntrs: ', cntrs)
        print(f'cntrs.shape[0]: ', cntrs.shape[0])
        if not cntrs.shape[0] and isinstance(output_dir, pathlib.Path):
            print(f'- Saving outlier image and mask to {output_dir}')
            output_dir_no_ptchs = output_dir / 'no_patches'
            os.makedirs(output_dir_no_ptchs, exist_ok=True)
            ts = get_ts()
            cv2.imwrite(str(output_dir_no_ptchs / f'img_{ts}.png'), image)
            cv2.imwrite(str(output_dir_no_ptchs / f'msk_{ts}.png'), mask)

    # print(f'Number of centroids: {len(cntrs)}')

    # - Create lists to store the crops
    img_objcts, msk_objcts = [], []

    for (x, y) in cntrs:
        img_crp, msk_crp = image[y:y+crop_height, x:x+crop_width], mask[y:y+crop_height, x:x+crop_width]
        # crp_w, crp_h = img_crp.shape[1], img_crp.shape[0]
        min_obj_pxls = (MIN_OBJ_PCT * crop_height * crop_width) // 100
        if msk_crp.sum() >= min_obj_pxls:
            # if (msk_crp.sum() >= min_obj_pxls) and (crp_w == crop_width and crp_h == crop_height):
            if img_crp.shape[0] != crop_height or img_crp.shape[1] != crop_width:
                img_crp = pad_image(img_crp, (crop_height, crop_width), pad_value=0)
                msk_crp = pad_image(msk_crp, (crop_height, crop_width), pad_value=0)

            if DEBUG_AUX_FUNCS and isinstance(output_dir, pathlib.Path):
                output_dir_valid_crps = output_dir / 'valid_crops'
                print(f'- Pixel count = {msk_crp.sum()} >= {min_obj_pxls}')
                print(f'- Saving valid crops to {output_dir_valid_crps}')
                os.makedirs(output_dir_valid_crps, exist_ok=True)
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(img_crp, cmap='gray')
                ax[1].imshow(msk_crp, cmap='gray')
                fig.savefig(output_dir_valid_crps / f'{get_ts()}.png')
                plt.close(fig)

            img_objcts.append(img_crp)
            msk_objcts.append(msk_crp)

        elif DEBUG_AUX_FUNCS and isinstance(output_dir, pathlib.Path):
            output_dir_low_obj_pxls = output_dir / 'low_obj_pxls'
            print(f'- Pixel count = {msk_crp.sum()} < {min_obj_pxls}')
            print(f'- Saving outlier crops to {output_dir_low_obj_pxls}')
            os.makedirs(output_dir_low_obj_pxls, exist_ok=True)
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(img_crp, cmap='gray')
            ax[1].imshow(msk_crp, cmap='gray')
            fig.savefig(output_dir_low_obj_pxls / f'{get_ts()}.png')
            plt.close(fig)

    return np.array(img_objcts, np.float32), np.array(msk_objcts, np.float32)


def connect_cells(mask: np.ndarray):
    # - Get the initial labels excluding the background
    lbls, lbl_px = np.unique(mask.astype(np.int8), return_counts=True)

    # - Clear low pixeled labels
    lbl_obj_idx = np.argwhere(lbl_px > 500)
    lbls = lbls[lbl_obj_idx]

    # - Clear the background label
    lbls = lbls[lbls > 0]

    # - Clean the mask from noise
    msk = np.zeros_like(mask, dtype=np.int8)
    for lbl in lbls:
        lbl_coords = np.argwhere(mask == lbl)
        lbl_x, lbl_y = lbl_coords[:, 0], lbl_coords[:, 1]
        msk[lbl_x, lbl_y] = lbl

    # Apply the Component analysis function
    (_, msk_cntd, stats, centroids) = cv2.connectedComponentsWithStats(msk.astype(np.uint8), cv2.CV_16U)

    # - For preserve the old labels
    msk_rpntd = np.zeros_like(msk_cntd, dtype=np.float32)

    # - Saves the labels to know if the labels was present
    lbl_cntd_roi_history = dict()
    for idx, lbl in enumerate(lbls):
        # TURN THE INSTANCE LABEL TO BINARY

        # - Copy the mask
        msk_bin = deepcopy(mask)

        # - Turn all the non-label pixels to 0
        msk_bin[msk_bin != lbl] = 0

        # - Turn all the label pixels to 1
        msk_bin[msk_bin > 0] = 1

        # - FIND THE CORRESPONDING CONNECTED COMPONENT IN THE CONNECTED
        # COMPONENT LABEL
        msk_cntd_roi = msk_bin * msk_cntd
        lbls_cntd_roi, n_pxs = np.unique(msk_cntd_roi, return_counts=True)
        lbls_cntd_roi, n_pxs = lbls_cntd_roi[lbls_cntd_roi > 0], n_pxs[1:]

        # REMOVE THE BACKGROUND LABEL
        # - Find the label with the maximum number of pixels in the ROI
        max_lbl_idx = np.argmax(n_pxs)

        # - Filter the labels with lower number of pixels in the ROI
        lbl_cntd_roi = lbls_cntd_roi[max_lbl_idx]

        # PAINT THE ROI
        msk_cntd_roi_bin = msk_cntd_roi / lbl_cntd_roi

        if lbl_cntd_roi not in lbl_cntd_roi_history.keys():
            # - If the color is new - paint it in this color and add it to the
            # history
            msk_rpntd += msk_cntd_roi_bin * lbl

            # - Add the ROI label to history
            lbl_cntd_roi_history[lbl_cntd_roi] = lbl
        else:
            # - If the color was previously used - the cells were connected,
            # so paint the ROI in the color previously used
            msk_rpntd += msk_cntd_roi_bin * lbl_cntd_roi_history.get(
                lbl_cntd_roi)

    return msk_rpntd


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
    bin_masks = []
    for lbl in lbls:
        bin_class_mask = deepcopy(inst_msk)
        bin_class_mask[bin_class_mask != lbl] = 0
        bin_class_mask[bin_class_mask > 0] = 1
        bin_masks.append(bin_class_mask)

    bin_masks = np.array(bin_masks)

    return bin_masks


def calc_seg_score(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Converts a multi-class label into a one-hot labels for each object in the
    multi-class label
    :param: multi_class_mask - mask where integers represent different objects
    """
    J = 0.0

    # - Ensure the multi-class label is populated with int values
    gt_mask = gt_mask.astype(np.int16)
    pred_mask = pred_mask.astype(np.int16)

    # - Find the classes
    lbls = np.unique(gt_mask)

    # - Discard the background (0)
    lbls = lbls[lbls > 0]
    if lbls.any():
        # - Convert the ground truth mask to one-hot class masks
        gt_one_hot_masks = split_instance_mask(instance_mask=gt_mask, labels=lbls)
        gt_one_hot_masks = np.array(gt_one_hot_masks, dtype=np.float32)

        # - Calculate the ground truth object area
        A_gt = gt_one_hot_masks.sum(axis=(-2, -1))

        # - Add another dimension for broadcasting
        if len(gt_one_hot_masks.shape) < 4:
            gt_one_hot_masks = np.expand_dims(gt_one_hot_masks, axis=0)

        # - Find the matching labels in the predicted label, as the number representing them may differ
        lbls = np.unique(gt_one_hot_masks * pred_mask)
        lbls = lbls[lbls > 0]

        # - Convert the predicted mask to one-hot class masks
        pred_one_hot_masks = split_instance_mask(instance_mask=pred_mask, labels=lbls)
        pred_one_hot_masks = np.array(pred_one_hot_masks, dtype=np.float32)

        # - Calculate the ground truth object area
        A_pred = pred_one_hot_masks.sum(axis=(-2, -1))

        # - Add another dimension for broadcasting
        if len(pred_one_hot_masks.shape) < 4:
            pred_one_hot_masks = np.expand_dims(pred_one_hot_masks, axis=0)

        # - Calculate an intersection for each object with each other object
        Is = (gt_one_hot_masks[:, np.newaxis, ...] * pred_one_hot_masks[np.newaxis, ...]).sum(axis=(-1, -2))

        # - Calculate the dice for each label for each mask
        J = (Is / (A_gt + A_pred - Is + EPSILON)).max(axis=1)

        # - Leave only the dice which are > 0.5 (may introduce np.inf in case all
        non_zero_iou_sums = (J > 0.5).sum(axis=0)

        # - Calculate the mean IoU for each mask
        J = J.sum(axis=0) / non_zero_iou_sums

        # - Replace all the dice which lower than 0.5 with 0
        J[(J == np.inf) | (np.isnan(J))] = .0

        # - Calculate the mean of the jaccards
        J = np.nanmean(J)

    return J


def update_hyper_parameters(hyper_parameters: dict, arguments: argparse.Namespace):

    # - Get hyper-parameter categories
    hyp_param_ctgrs = list(hyper_parameters.keys())

    # - Get the argument names
    args = vars(arguments)
    arg_names = list(args.keys())

    if args.get('arch1'):
        hyper_parameters.get('model')['architecture'] = hyper_parameters.get('model')['architecture1']

    elif args.get('arch2'):
        hyper_parameters.get('model')['architecture'] = hyper_parameters.get('model')['architecture2']

    elif args.get('arch3'):
        hyper_parameters.get('model')['architecture'] = hyper_parameters.get('model')['architecture3']

    # - For each argument
    for arg_name in arg_names:
        hyp_param = args.get(arg_name)
        if hyp_param is not None:
            for hyp_param_ctgr in hyp_param_ctgrs:
                # - Hyper-parameter update with the parameter in args
                hyper_parameters.get(hyp_param_ctgr)[arg_name] = hyp_param


def normalize_image(image: np.ndarray):
    return (image - image.mean() * np.max(image)) / (image.std() * np.max(image))


def calc_histogram(data: np.ndarray or list, bins: np.ndarray, normalize: bool = False):
    # - Plot histogram
    heights, ranges = np.histogram(data, range=(bins[0], bins[-1]),
                                   bins=len(bins), density=False)

    if normalize:
        heights /= len(data)

    return heights, ranges


def to_numpy(data: np.ndarray, file_path: str or pathlib.Path, overwrite: bool = False, logger: logging.Logger = None):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if isinstance(file_path, pathlib.Path):
        if overwrite or not file_path.is_file():
            os.makedirs(file_path.parent, exist_ok=True)
            np.save(str(file_path), data)
        else:
            warning_log(logger=logger,
                        message=f'Could not save the data to the file '
                                f'\'{file_path}\' as it already exists '
                                f'(set overwrite=True to save on existing '
                                f'files in the future)! ')
    else:
        warning_log(logger=logger,
                    message=f'Could not save the data to the file '
                            f'\'{file_path}\' - \'file_path\' must be in '
                            f'format str or pathlib.Path but is of type '
                            f'\'{type(file_path)}\'!')


def from_numpy(file_path: str or pathlib.Path, logger: logging.Logger = None):
    data = None

    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if isinstance(file_path, pathlib.Path):
        if file_path.is_file():
            data = np.load(str(file_path))
        else:
            warning_log(logger=logger, message=f'Could not load the data from '
                                               f'\'{file_path}\' file - the '
                                               f'file does not exists!')
    else:
        warning_log(logger=logger, message=f'Could not load the data from '
                                           f'\'{file_path}\' file - the file '
                                           f'must be of type str or '
                                           f'pathlib.Path, but is of type '
                                           f'\'{type(file_path)}\'!')

    return data


def from_pickle(data_file: pathlib.Path or str, logger: logging.Logger = None):
    data = None
    if check_pathable(path=data_file):
        data_file = str_2_path(path=data_file)

        data = pkl.load(data_file.open(mode='rb'))
    else:
        warning_log(logger=logger, message=f'Could not load the data from file '
                                           f'\'{data_file}\'!')

    return data


def to_pickle(data, save_file: str or pathlib.Path, logger: logging.Logger = None):
    """
    :param: data
    """
    if check_pathable(path=save_file):
        save_file = str_2_path(path=save_file)

        pkl.dump(data, save_file.open(mode='wb'))
    else:
        warning_log(logger=logger, message=f'Could not save the data to file '
                                           f'\'{save_file}\'!')


def print_pretty_message(message: str, delimiter_symbol: str = '='):
    """
    :param: message
    :param: delimiter_symbol
    """
    delimiter_len = len(message) + 6

    print('\n\t', end='')
    for i in range(delimiter_len):
        print(delimiter_symbol, end='')

    print(f'\n\t{delimiter_symbol}{delimiter_symbol} {message} '
          f'{delimiter_symbol}{delimiter_symbol}')

    print('\t', end='')
    for i in range(delimiter_len):
        print(delimiter_symbol, end='')

    print('')


def get_metrics(x: np.ndarray, y: np.ndarray):

    # - Calculate pearson correlation
    rho, p = pearsonr(x, y)

    # - Calculate mean squared error
    mse = np.mean(np.square(x - y))

    return rho, p, mse


def get_data(mode: str, hyper_parameters: dict, logger: logging.Logger = None):
    dt_fl = str_2_path(path=hyper_parameters.get(mode)['temp_data_file'])
    if not hyper_parameters.get('data')['reload_data'] and dt_fl.is_file():
        data_dict = from_pickle(data_file=dt_fl, logger=logger)
    else:
        dt_dir = str_2_path(path=hyper_parameters.get(mode)['data_dir'])
        img_msk_fl_tpls = get_image_mask_file_tuples(
            root_dir=dt_dir,
            seg_dir_postfix=hyper_parameters.get(mode)['seg_dir_postfix'],
            image_prefix=hyper_parameters.get(mode)['image_prefix'],
            seg_prefix=hyper_parameters.get(mode)['seg_prefix'],
            seg_sub_dir=hyper_parameters.get(mode)['seg_sub_dir']
        )

        # - Load images and their masks
        if mode == 'training':
            np.random.shuffle(img_msk_fl_tpls)
            data_dict = get_data_dict(image_mask_file_tuples=img_msk_fl_tpls)

            # - Clean data items with no objects in them
            data_dict = clean_items_with_empty_masks(
                data_dict=data_dict,
                save_file=hyper_parameters.get(mode)['temp_data_file'])

            # data_dict = get_relevant_data(
            #     data_dict=data_dict,
            #     relevant_files=os.listdir(
            #         hyper_parameters.get(mode)['mask_dir']),
            #     save_file=hyper_parameters.get(mode)['temp_data_file'],
            #     logger=logger)
        else:
            data_dict = get_data_dict(image_mask_file_tuples=img_msk_fl_tpls)

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

    parser.add_argument('--debug', default=False, action='store_true', help=f'If the run is a debugging run')
    parser.add_argument('--gpu_id', type=int, default=-1, help='The ID of the GPU to run on')

    # - GENERAL PARAMETERS
    parser.add_argument('--project', type=str, help='The name of the project')
    parser.add_argument('--name', type=str, help='The name of the experiment')
    parser.add_argument('--queue_name', type=str, help='The name of the queue to assign the task to')
    parser.add_argument('--local_execution', default=False, action='store_true', help=f'If to execute the task locally')
    parser.add_argument('--arch1', default=False, action='store_true', help=f'''
    > arch1: 
        conv2d_blocks:
          out_channels: [64, 128, 256, 256]
          kernel_sizes: [5, 5, 5, 5]
          dropblock_rate: 0.1
          dropblock_size: 7

        fc_blocks:
          out_features: [512, 1024]
          drop_rate: 0.2
    ''')

    parser.add_argument('--arch2', default=False, action='store_true', help=f'''
    > arch2: 
        conv2d_blocks:
          out_channels: [64, 128, 256, 256]
          kernel_sizes: [3, 3, 3, 3]
          dropblock_rate: 0.1
          dropblock_size: 7
          
        fc_blocks:
          out_features: [128, 128]
          drop_rate: 0.5
    ''')

    parser.add_argument('--arch3', default=False, action='store_true', help=f'''
    > arch3:
        conv2d_blocks:
          out_channels: [32, 64, 128, 256]
          kernel_sizes: [5, 5, 5, 5]
          dropblock_rate: 0.1
          dropblock_size: 7
          
        fc_blocks:
          out_features: [512, 1024]
          drop_rate: 0.2
    ''')

    # parser.add_argument('--architecture', type=str, choices=['arch1', 'arch2', 'arch3'], help=f'''
    # > arch1:
    #     conv2d_blocks:
    #       out_channels: [64, 128, 256, 256]
    #       kernel_sizes: [5, 5, 5, 5]
    #       dropblock_rate: 0.1
    #       dropblock_size: 7
    #
    #     fc_blocks:
    #       out_features: [512, 1024]
    #       drop_rate: 0.2
    #
    # > arch2:
    #     conv2d_blocks:
    #       out_channels: [64, 128, 256, 256]
    #       kernel_sizes: [3, 3, 3, 3]
    #       dropblock_rate: 0.1
    #       dropblock_size: 7
    #
    #     fc_blocks:
    #       out_features: [128, 128]
    #       drop_rate: 0.5
    #
    # > arch3:
    #     conv2d_blocks:
    #       out_channels: [32, 64, 128, 256]
    #       kernel_sizes: [5, 5, 5, 5]
    #       dropblock_rate: 0.1
    #       dropblock_size: 7
    #
    #     fc_blocks:
    #       out_features: [512, 1024]
    #       drop_rate: 0.2
    #                     ''')

    parser.add_argument('--output_dir', type=str, help='The path to the directory where the outputs will be placed')
    parser.add_argument('--hyper_params_file', type=str, default=HYPER_PARAMS_FILE,
                        help=f'The path to the file with the hyper-parameters')
    parser.add_argument('--checkpoint_file', type=str, help=f'The path to the file which contains the '
                                                            f'checkpoints of the model')
    parser.add_argument('--checkpoint_dir', type=str, help=f'The path to the directory which contains the '
                                                           f'checkpoints of the model')

    # - Train
    parser.add_argument('--learning_rate_optimization_epochs', type=int,
                        help=f'The number of optimization epochs for the learning rate')
    parser.add_argument('--reload_data', default=False, action='store_true', help=f'If to reload the data')
    parser.add_argument('--train_data_dir', type=str, help='The path to the train data file')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, help='The number of samples in each train batch')
    parser.add_argument('--val_batch_size', type=int, help='The number of samples in each validation batch')
    parser.add_argument('--optimizer', type=str,
                        choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'],
                        help=f'The optimizer to use')
    parser.add_argument('--weighted_loss', default=False, action='store_true',
                        help=f'If to use the weighted version of the MSE loss')
    parser.add_argument('--load_checkpoint', default=False, action='store_true',
                        help=f'If to continue the training from the checkpoint saved at the checkpoint file')
    parser.add_argument('--learning_rate', type=float, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--kernel_regularizer_type', type=str,
                        choices=['l1', 'l2', 'l1l2'],
                        help=f'The type of the regularization')
    parser.add_argument('--crop_height', type=int,
                        help='The height of the images that will be used for network training and inference. '
                             'If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')
    parser.add_argument('--crop_width', type=int,
                        help='The width of the images that will be used for network training and inference. '
                             'If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')

    # - Callbacks
    parser.add_argument('--wandb', default=False, action='store_true', help=f'If to use the wandb callback')
    parser.add_argument('--no_tensorboard', default=False, action='store_true',
                        help=f'If to use the tensorboard callback')

    # - Test flags
    parser.add_argument('--test_data_dir', type=str, help='The path to the test data dir')
    parser.add_argument('--test_sim', default=False, action='store_true', help=f'Run test on the SIM+ data')
    parser.add_argument('--test_gowt1', default=False, action='store_true', help=f'Run test on the GWOT1 data')
    parser.add_argument('--test_hela', default=False, action='store_true', help=f'Run test on the HELA data')
    parser.add_argument('--test_all', default=False, action='store_true',
                        help=f'Run test on all the test data sets')

    # - Inference
    parser.add_argument('--inference_data_dir', type=str, help='The path to the inference data dir')
    parser.add_argument('--infer_all', default=False, action='store_true',
                        help=f'Run inference on all the test data sets')

    # > SIM+ Data
    parser.add_argument('--infer_bgu_3_sim', default=False, action='store_true',
                        help=f'Run inference on the SIM+ data by BGU-IL(3) model')
    parser.add_argument('--infer_cvut_sim', default=False, action='store_true',
                        help=f'Run inference on the SIM+ data by CVUT-CZ model')
    parser.add_argument('--infer_kth_sim', default=False, action='store_true',
                        help=f'Run inference on the SIM+ data by KTH-SE model')
    parser.add_argument('--infer_unsw_sim', default=False, action='store_true',
                        help=f'Run inference on the SIM+ data by UNSW-AU model')
    parser.add_argument('--infer_dkfz_sim', default=False, action='store_true',
                        help=f'Run inference on the SIM+ data by DKFZ-GE model')

    # > GWOT1 Data
    parser.add_argument('--infer_bgu_4_gowt1', default=False, action='store_true',
                        help=f'Run inference on the GWOT1 data by BGU-IL(4) model')
    parser.add_argument('--infer_bgu_5_gowt1', default=False, action='store_true',
                        help=f'Run inference on the GWOT1 data by BGU-IL(5) model')
    parser.add_argument('--infer_unsw_gowt1', default=False, action='store_true',
                        help=f'Run inference on the GWOT1 data by UNSW-AU model')
    parser.add_argument('--infer_kth_gowt1', default=False, action='store_true',
                        help=f'Run inference on the GWOT1 data by KTH-SE model')

    return parser


def instance_2_categorical(instance_masks: np.ndarray or list):
    """
    Converts an instance masks (i.e., where each cell is represented by a
    different color, to a mask with 3 classes, i.e.,
        - 0 for background,
        - 1 for the inner part of the cell
        - 2 for cells' boundary
    """

    def _merge_categorical_masks(masks: np.ndarray):
        msk = np.zeros(masks.shape[1:], dtype=np.float32)
        for cat_mks in masks:
            msk += cat_mks

        # - Fix the boundary if it was added several times from different cells
        msk[msk > 2.] = 2.
        msk = msk.astype(np.float32)

        return msk

    def _get_categorical_mask(binary_mask: np.ndarray):
        # Shrinks the labels
        inner_msk = grey_erosion(binary_mask, size=3)

        # Create the contur of the cells
        contur_msk = binary_mask - inner_msk
        contur_msk[contur_msk > 0] = 2

        # - Create the inner part of the cell
        inner_msk[inner_msk > 0] = 1

        # - Combine the inner and the contur masks to create the categorical mask
        # with three classes, i.e., background 0, inner 1 and contur 2
        cat_msk = inner_msk + contur_msk

        return cat_msk

    btch_cat_msks = []
    for mask in instance_masks:
        # - Split the mask into binary masks of separate cells
        bin_msks = split_instance_mask(instance_mask=mask)

        # - For each binary cell - get a categorical mask
        categorical_masks = [np.zeros_like(mask)]
        for bin_msk in bin_msks:
            categorical_msk = _get_categorical_mask(binary_mask=bin_msk)
            categorical_masks.append(categorical_msk)

        # - Merge the categorical masks for each cell into a single
        # categorical mask
        mrgd_cat_msk = _merge_categorical_masks(masks=np.array(categorical_masks))

        # - Add the categorical mask to the batch masks
        btch_cat_msks.append(mrgd_cat_msk)

    # - Convert to array
    btch_cat_msks = np.array(btch_cat_msks, dtype=np.float32)

    return btch_cat_msks


def test_instance_2_categorical_single():
    msk_57_fl = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-N2DH-SIM+/01_RES/mask057.tif'

    msk_57_inst = load_image(msk_57_fl)

    msk_57_cat = instance_2_categorical(instance_masks=[msk_57_inst])[0]
    msk_57_cat = categorical_2_rgb(msk_57_cat)

    plt.imshow(msk_57_inst)
    plt.show()
    plt.imshow(msk_57_cat)
    plt.show()


def test_instance_2_categorical_batch():
    msk_57_fl = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-N2DH-SIM+/01_RES/mask057.tif'
    msk_77_fl = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-N2DH-SIM+/01_RES/mask077.tif'
    msk_91_fl = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-N2DH-SIM+/01_RES/mask091.tif'

    msk_57_inst = load_image(msk_57_fl)
    msk_77_inst = load_image(msk_77_fl)
    msk_91_inst = load_image(msk_91_fl)

    msk_57_cat, msk_77_cat, msk_91_cat = instance_2_categorical(instance_masks=[msk_57_inst, msk_77_inst, msk_91_inst])

    fig, ax = plt.subplots(2, 3, figsize=(60, 20))
    ax[0, 0].imshow(msk_57_inst)
    ax[1, 0].imshow(msk_57_cat)

    ax[0, 1].imshow(msk_77_inst)
    ax[1, 1].imshow(msk_77_cat)

    ax[0, 2].imshow(msk_91_inst)
    ax[1, 2].imshow(msk_91_cat)

    plt.show()


if __name__ == '__main__':
    test_instance_2_categorical_single()
