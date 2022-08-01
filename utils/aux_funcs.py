import argparse
import logging.config
import pickle as pkl

import torch
import yaml

import logging
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from configs.general_configs import (
    TRAIN_DATA_FILE,
    INFERENCE_DATA_FILE,
    TEST_GT_DATA_FILE,
    TEST_ST_DATA_FILE,
    OUTPUT_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IN_CHANNELS,
    OUT_CHANNELS,
    EPOCHS,
    TRAIN_BATCH_SIZE,
    NUM_WORKERS,
    VAL_PROP,
    CHECKPOINT_FILE,
    OPTIMIZER_LR,
    OPTIMIZER_LR_DECAY,
    OPTIMIZER_BETA_1,
    OPTIMIZER_BETA_2,
    OPTIMIZER_RHO,
    OPTIMIZER_WEIGHT_DECAY,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_DAMPENING,
    OPTIMIZER_MOMENTUM_DECAY,
    OPTIMIZER, DEFAULT_MODEL_LIB
)
plt.style.use('seaborn')


def check_dir(path: str or pathlib.Path):
    dir_valid = False

    # - Convert path to pathlib.Path object
    if isinstance(path, str):
        dir_path = pathlib.Path(path)

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


def err_log(logger: logging.Logger, message: str):
    if isinstance(logger, logging.Logger):
        logger.exception(message)
    else:
        print(message)


def data_not_found_err(data_file: str or pathlib.Path, logger: logging.Logger):
    err_log(logger=logger, message=f'Could not load data from \'{data_file}\'!')


def get_model_configs(configs_file: pathlib.Path, logger: logging.Logger):
    if configs_file.is_file():
        model_configs = read_yaml(configs_file)
        if model_configs is not None:
            info_log(logger=logger, message=f'The model configs were successfully loaded from \'{configs_file}\'!')
        else:
            info_log(logger=logger, message=f'The model configs file ({configs_file}) does not contain valid model configurations!')
    else:
        info_log(logger=logger, message=f'The model configs file ({configs_file}) does not exist!')
    return model_configs


def scan_files(root_dir: pathlib.Path or str, seg_dir_postfix: str, image_prefix: str, seg_prefix: str):
    file_tuples = list()
    for root, dirs, _ in os.walk(root_dir):
        root = pathlib.Path(root)
        for dir in dirs:
            seg_dir = root / f'{dir}_{seg_dir_postfix}'
            if seg_dir.is_dir():
                for sub_root, _, files in os.walk(root / dir):
                    for file in files:
                        img_fl = pathlib.Path(f'{sub_root}/{file}')
                        seg_fl = pathlib.Path(f'{seg_dir}/{file.replace(image_prefix, seg_prefix)}')
                        if img_fl.is_file() and seg_fl.is_file():
                            file_tuples.append((img_fl, seg_fl))
    return file_tuples


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


def calc_jaccard(R: np.ndarray, S: np.ndarray):
    """
    Calculates the mean Jaccard coefficient for two multi-class labels
    :param: R - Reference multi-class label
    :param: S - Segmentation multi-class label
    """
    def _get_one_hot_masks(multi_class_mask: np.ndarray, classes: np.ndarray = None):
        """
        Converts a multi-class label into a one-hot labels for each object in the multi-class label
        :param: multi_class_mask - mask where integers represent different objects
        """
        # - Ensure the multi-class label is populated with int values
        mlt_cls_mask = multi_class_mask.astype(np.int16)

        # - Find the classes
        cls = classes
        if cls is None:
            cls = np.unique(mlt_cls_mask)

        # - Discard the background (0)
        cls = cls[cls > 0]

        one_hot_masks = np.zeros((len(cls), *mlt_cls_mask.shape), dtype=np.float32)
        for lbl_idx, lbl in enumerate(one_hot_masks):
            # for obj_mask_idx, _ in enumerate(lbl):
            idxs = np.argwhere(mlt_cls_mask == cls[lbl_idx])
            x, y = idxs[:, 0], idxs[:, 1]
            one_hot_masks[lbl_idx, x, y] = 1.
        return one_hot_masks

    # - In case theres no other classes besides background (i.e., 0) J = 0.
    J = 0.
    R = np.array(R)
    S = np.array(S)
    # - Convert the multi-label mask to multiple one-hot masks
    cls = np.unique(R.astype(np.int16))

    if cls[cls > 0].any():  # <= If theres any classes besides background (i.e., 0)
        R = _get_one_hot_masks(multi_class_mask=R, classes=cls)
        S = _get_one_hot_masks(multi_class_mask=S, classes=cls)

        # - Calculate the intersection of R and S
        I_sums = np.sum(R[:, np.newaxis, ...] * S[np.newaxis, ...], axis=(-2, -1))
        x, y = np.arange(len(I_sums)), np.argmax(I_sums, axis=1)  # <= Choose the once that have the largest overlap with the ground truth label
        I = I_sums[x, y]

        # - Calculate the union of R and S
        U_sums = np.sum(np.logical_or(R[:, np.newaxis, ...], S[np.newaxis, ...]), axis=(-2, -1))
        U = U_sums[x, y]

        # - Mean Jaccard on the valid items only
        U[U <= 0] = 1  # <= To avoid division by 0
        J = (I / U)

        # - Calculate the areas of the reference items
        R_areas = R.sum(axis=(-2, -1))
        R_areas[R_areas <= 0] = 1  # <= To avoid division by 0

        # - Find out the indices of the items which do not satisfy |I| / |R| > 0.5 and replace them with 0
        inval = np.argwhere((I / R_areas) <= .5).reshape(-1)

        J[inval] = 0.0

        # J = J[J > 0].mean() if J[J > 0].any() else 0.0

    return J.mean() if isinstance(J, np.ndarray) else J


def plot_hist(data: np.ndarray or list, hist_range: list or tuple, bins: int, save_name: str, output_dir: pathlib.Path, density: bool = True):
    # - Plot histogram
    heights, ranges = np.histogram(data, range=hist_range[:-1], bins=bins, density=density)
    fig, ax = plt.subplots()
    ax.bar(ranges[:-1], heights)
    ax.set(xticks=np.arange(hist_range[0], hist_range[1]+hist_range[2], hist_range[2]), xlim=[0, 1])
    plt.savefig(output_dir / f'{save_name}.png')

    plt.close(fig)


def to_pickle(file, name: str, save_dir: str or pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    pkl.dump(file, (save_dir / (name + '.pkl')).open(mode='wb'))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--model_lib', type=str, choices=['pytorch', 'tf'], default=DEFAULT_MODEL_LIB, help=f'The library used to build the model (pytorch or tensorflow)')
    parser.add_argument('--load_model', default=False, action='store_true', help=f'If to load the model')
    parser.add_argument('--train', default=False, action='store_true', help=f'If to run the train procedure')
    parser.add_argument('--infer', default=False, action='store_true', help=f'If to run the inference procedure')
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, torch.cuda.device_count() - 1)], default=0 if torch.cuda.device_count() > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')

    parser.add_argument('--train_continue', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at \'{CHECKPOINT_FILE}\'')
    parser.add_argument('--train_data_file', type=str, default=TRAIN_DATA_FILE, help='The path to the train data file')
    parser.add_argument('--inference_data_file', type=str, default=INFERENCE_DATA_FILE, help='The path to the inference data file')
    parser.add_argument('--test_gt_data_file', type=str, default=TEST_GT_DATA_FILE, help='The path to the gold standard test file')
    parser.add_argument('--test_st_data_file', type=str, default=TEST_ST_DATA_FILE, help='The path to the silver standard test file')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH, help='The width of the images that will be used for network training and inference. If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT, help='The height of the images that will be used for network training and inference. If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')

    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help='The number of channels in an input image (e.g., 3 for RGB, 1 for Grayscale etc)')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS, help='The number of channels in the output image (e.g., 3 for RGB, 1 for Grayscale etc)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=TRAIN_BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='The number of workers to load the data')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--checkpoint_file', type=str, default=CHECKPOINT_FILE, help=f'The path to the file which contains the checkpoints of the model')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'], default=OPTIMIZER,  help=f'The optimizer to use')

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


    return parser
