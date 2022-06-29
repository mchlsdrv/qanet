import os
import re
from functools import partial

import yaml
import pickle as pkl
import logging
import logging.config
import threading
import multiprocessing as mlp
import argparse
import pathlib
import tensorflow as tf
import numpy as np
from custom.models import (
    RibCage
)

from custom.callbacks import (
    ProgressLogCallback
)
from configs.general_configs import (
    IMAGE_SIZE,

    TRAIN_DIR,

    TEST_DIR,

    INFERENCE_DIR,

    OUTPUT_DIR,

    EPOCHS,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    VALIDATION_PROPORTION,
    LEARNING_RATE,

    TENSOR_BOARD,
    TENSOR_BOARD_WRITE_IMAGES,
    TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
    TENSOR_BOARD_UPDATE_FREQ,

    PROGRESS_LOG,
    PROGRESS_LOG_INTERVAL,
    SCATTER_PLOT_FIGSIZE,

    TENSOR_BOARD_LAUNCH,

    EARLY_STOPPING,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EARLY_STOPPING_VERBOSE,

    TERMINATE_ON_NAN,

    REDUCE_LR_ON_PLATEAU,
    REDUCE_LR_ON_PLATEAU_MONITOR,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE,
    REDUCE_LR_ON_PLATEAU_MIN_DELTA,
    REDUCE_LR_ON_PLATEAU_COOLDOWN,
    REDUCE_LR_ON_PLATEAU_MIN_LR,
    REDUCE_LR_ON_PLATEAU_MODE,
    REDUCE_LR_ON_PLATEAU_VERBOSE,

    MODEL_CHECKPOINT,
    MODEL_CHECKPOINT_FILE_BEST_MODEL_TEMPLATE,
    MODEL_CHECKPOINT_MONITOR,
    MODEL_CHECKPOINT_VERBOSE,
    MODEL_CHECKPOINT_SAVE_BEST_ONLY,
    MODEL_CHECKPOINT_MODE,
    MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY,
    MODEL_CHECKPOINT_SAVE_FREQ, CHECKPOINT_DIR, OPTIMIZER, OPTIMIZER_RHO, OPTIMIZER_BETA_1, OPTIMIZER_BETA_2, OPTIMIZER_MOMENTUM, KERNEL_REGULARIZER_TYPE, KERNEL_REGULARIZER_L1, KERNEL_REGULARIZER_L2, KERNEL_REGULARIZER_FACTOR, KERNEL_REGULARIZER_MODE, ACTIVATION, ACTIVATION_LEAKY_RELU_ALPHA, ACTIVATION_RELU_NEGATIVE_SLOPE, ACTIVATION_RELU_THRESHOLD, ACTIVATION_RELU_MAX_VALUE, CLAHE_TILE_GRID_SIZE, CLAHE_CLIP_LIMIT,
)

# from utils.visualisation_utils.plotting_funcs import (
from utils.plotting_funcs import (
    plot_scatter
)

# from utils.image_utils.image_aux import (
from utils.image_funcs import (
    get_image_from_figure
)


def decode_file(file):
    if isinstance(file, bytes):
        file = file.decode('utf-8')
    return file


# noinspection PyTypeChecker
def get_callbacks(callback_type: str, output_dir: pathlib.Path, logger: logging.Logger = None):
    callbacks = []
    # -------------------
    # Built-in  callbacks
    # -------------------
    tb_prc = None
    if TENSOR_BOARD:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=output_dir,
                write_images=TENSOR_BOARD_WRITE_IMAGES,
                write_steps_per_second=TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
                update_freq=TENSOR_BOARD_UPDATE_FREQ,
            )
        )
        if PROGRESS_LOG:
            callbacks.append(
                ProgressLogCallback(
                    log_type=callback_type,
                    figsize=SCATTER_PLOT_FIGSIZE,
                    log_dir=output_dir,
                    log_interval=PROGRESS_LOG_INTERVAL,
                    logger=logger
                )
            )
        # - Launch the tensorboard in a thread
        if TENSOR_BOARD_LAUNCH:
            info_log(logger=logger, message=f'Launching a Tensor Board thread on logdir: \'{output_dir}\'...')
            tb_prc = mlp.Process(
                target=lambda: os.system(f'tensorboard --logdir={output_dir}'),
            )

    if EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=EARLY_STOPPING_MONITOR,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                patience=EARLY_STOPPING_PATIENCE,
                mode=EARLY_STOPPING_MODE,
                restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
                verbose=EARLY_STOPPING_VERBOSE,
            )
        )

    if TERMINATE_ON_NAN:
        callbacks.append(
            tf.keras.callbacks.TerminateOnNaN()
        )

    if REDUCE_LR_ON_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=REDUCE_LR_ON_PLATEAU_MONITOR,
                factor=REDUCE_LR_ON_PLATEAU_FACTOR,
                patience=REDUCE_LR_ON_PLATEAU_PATIENCE,
                min_delta=REDUCE_LR_ON_PLATEAU_MIN_DELTA,
                cooldown=REDUCE_LR_ON_PLATEAU_COOLDOWN,
                min_lr=REDUCE_LR_ON_PLATEAU_MIN_LR,
                mode=REDUCE_LR_ON_PLATEAU_MODE,
                verbose=REDUCE_LR_ON_PLATEAU_VERBOSE,
            )
        )

    if MODEL_CHECKPOINT:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir / MODEL_CHECKPOINT_FILE_BEST_MODEL_TEMPLATE,
                monitor=MODEL_CHECKPOINT_MONITOR,
                verbose=MODEL_CHECKPOINT_VERBOSE,
                save_best_only=MODEL_CHECKPOINT_SAVE_BEST_ONLY,
                mode=MODEL_CHECKPOINT_MODE,
                save_weights_only=MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY,
                save_freq=MODEL_CHECKPOINT_SAVE_FREQ,
            )
        )

    return callbacks, tb_prc


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


def get_train_val_idxs(n_items, val_prop):
    all_idxs = np.arange(n_items)
    val_idxs = np.random.choice(all_idxs, int(val_prop * n_items), replace=False)
    train_idxs = np.setdiff1d(all_idxs, val_idxs)
    return train_idxs, val_idxs


def get_one_hot_masks(multi_class_mask: np.ndarray, classes: np.ndarray = None):
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


def calc_jaccard(R: np.ndarray, S: np.ndarray):
    """
    Calculates the mean Jaccard coefficient for two multi-class labels
    :param: R - Reference multi-class label
    :param: S - Segmentation multi-class label
    """
    # - In case theres no other classes besides background (i.e., 0) J = 0.
    J = 0.

    # - Convert the multi-label mask to multiple one-hot masks
    cls = np.unique(R.astype(np.int16))

    if cls[cls > 0].any():  # <= If theres any classes besides background (i.e., 0)
        R = get_one_hot_masks(multi_class_mask=R, classes=cls)
        S = get_one_hot_masks(multi_class_mask=S, classes=cls)

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

        J[inval] = np.nan

        J = np.nanmean(J)

        J = np.nan_to_num(J, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    return J


def check_dir(dir_path: pathlib.Path):
    dir_exists = False
    if isinstance(dir_path, pathlib.Path):
        if not dir_path.is_dir():
            os.makedirs(dir_path)
            dir_exists = True
    return dir_exists


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def launch_tensorboard(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th


def get_model(model_configs, checkpoint_dir: pathlib.Path = None, logger: logging.Logger = None):
    weights_loaded = False

    model = RibCage(model_configs=model_configs, logger=logger)

    if checkpoint_dir.is_dir:
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_cpt is not None:
                model.load_weights(latest_cpt)
                weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                logger.exception(f'Can\'t load weighs from \'{checkpoint_dir}\' due to error: {err}')
        else:
            if isinstance(logger, logging.Logger):
                if latest_cpt is not None:
                    logger.info(f'Weights from \'{checkpoint_dir}\' were loaded successfully to the \'RibCage\' model!')
                else:
                    logger.info(f'No weights were found to load in \'{checkpoint_dir}\'!')
    if isinstance(logger, logging.Logger):
        logger.info(model.summary())

    return model, weights_loaded


def get_optimizer(algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'adam':
        optimizer = partial(
            tf.keras.optimizers.Adam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
            amsgrad=args.get('amsgrad'),
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            tf.keras.optimizers.Nadam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            tf.keras.optimizers.Adamax,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad
    elif algorithm == 'adadelta':
        optimizer = partial(
            tf.keras.optimizers.Adadelta,
            rho=args.get('rho'),
        )
    elif algorithm == 'sgd':
        optimizer = partial(
            tf.keras.optimizers.SGD,
            momentum=args.get('momentum'),
            nesterov=args.get('nesterov'),
        )
    elif algorithm == 'rms_prop':
        optimizer = partial(
            tf.keras.optimizers.RMSprop,
            rho=args.get('rho'),
            momentum=args.get('momentum'),
            centered=args.get('centered'),
        )
    return optimizer(learning_rate=args.get('learning_rate'))


def choose_gpu(gpu_id: int = 0, logger: logging.Logger = None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_id > -1:
                # print(device_lib.list_local_devices())
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                # print(device_lib.list_local_devices())
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on: {logical_gpus} (GPU #{gpu_id})
                ====================================================
                ''')
            else:
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on all available devices
                ====================================================
                    ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)


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


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--data_from_single_dir', default=False, action='store_true', help='If the data should be taken from a single directory, or collected from several directories')

    parser.add_argument('--reload_data', default=False, action='store_true', help=f'If to reload data from files and overwrite the temp data')

    parser.add_argument('--train', default=False, action='store_true', help=f'If to perform the train of the current network')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR, help='The path to the top directory where the images and the segmentations are stored')

    parser.add_argument('--test', default=False, action='store_true', help=f'If to perform the test of the current network')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help=f'Path to the test images, their segmentations and a file with the corresponding seg measures. Images should be placed in a folder called \'imgs\', the segmentations in a folder called \'segs\', and the file with the seg measures should be called \'seg_measures.pkl\', and be placed together with the two previous folders')

    parser.add_argument('--inference', default=False, action='store_true', help=f'If to perform the inference with the current network')
    parser.add_argument('--inference_dir', type=str, default=INFERENCE_DIR, help=f'Path to the images to infer. Images should be placed in a folder called \'imgs\', the segmentations in a folder called \'segs\', and the file with the seg measures should be called \'seg_measures.pkl\', and be placed together with the two previous folders')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help=f'The path to the directory that contains the checkpoints of the model')

    # - AUGMENTATIONS
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')
    parser.add_argument('--clahe_clip_limit', type=float, default=CLAHE_CLIP_LIMIT, help='The size of the CLAHE filter clip limit by which it thresholds the values')
    parser.add_argument('--clahe_tile_grid_size', type=int, default=CLAHE_TILE_GRID_SIZE, help='The size of the CLAHE filter size')

    # - TRAINING CONFIGS
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE, help='The number of samples in each validation batch')
    parser.add_argument('--validation_proportion', type=float, default=VALIDATION_PROPORTION, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help=f'The initial learning rate of the optimizer')

    # - OPTIMIZER
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'nadam', 'adadelta', 'adamax', 'adagrad', 'rms_prop'], default=OPTIMIZER,  help=f'The optimizer to use')
    parser.add_argument('--optimizer_rho', type=float, default=OPTIMIZER_RHO, help=f'The decay rate (Adadelta, RMSprop)')
    parser.add_argument('--optimizer_beta_1', type=float, default=OPTIMIZER_BETA_1, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, default=OPTIMIZER_BETA_2, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_momentum', type=float, default=OPTIMIZER_MOMENTUM, help=f'The momentum (SGD, RMSprop)')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    # - ACTIVATION
    parser.add_argument('--activation', type=str, choices=['swish', 'relu', 'leaky_relu'], default=ACTIVATION,  help=f'The activation to use')
    parser.add_argument('--activation_relu_max_value', type=float, default=ACTIVATION_RELU_MAX_VALUE,  help=f'The negative slope in the LeakyReLU activation function for values < 0')
    parser.add_argument('--activation_relu_negative_slope', type=float, default=ACTIVATION_RELU_NEGATIVE_SLOPE,  help=f'The negative slope in the ReLU activation function for values < 0')
    parser.add_argument('--activation_relu_threshold', type=float, default=ACTIVATION_RELU_THRESHOLD, help=f'The value that has to be exceeded activate the neuron')
    parser.add_argument('--activation_leaky_relu_alpha', type=float, default=ACTIVATION_LEAKY_RELU_ALPHA,  help=f'The negative slope in the LeakyReLU activation function for values < 0')

    # - DROP BLOCK
    parser.add_argument('--drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, help=f'The size of the block to drop')

    # - KERNEL REGULARIZER
    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], default=KERNEL_REGULARIZER_TYPE, help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, default=KERNEL_REGULARIZER_L1, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, default=KERNEL_REGULARIZER_L2, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, default=KERNEL_REGULARIZER_FACTOR, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], default=KERNEL_REGULARIZER_MODE, help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    return parser


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


def write_scalars_to_tensorboard(writer, data: dict, step: int):
    with writer.as_default():
        with tf.device('/cpu:0'):
            # SCALARS
            # - Write the loss
            tf.summary.scalar(
                'Loss',
                data.get('Loss'),
                step=step
            )


def write_images_to_tensorboard(writer, data: dict, step: int):
    with writer.as_default():
        with tf.device('/cpu:0'):
            # - Write the images
            # -> Normalize the images
            imgs = data.get('Images')
            disp_imgs = imgs - tf.reduce_min(imgs, axis=(1, 2, 3), keepdims=True)
            disp_imgs = disp_imgs / tf.reduce_max(disp_imgs, axis=(1, 2, 3), keepdims=True)
            tf.summary.image(
                'Images',
                disp_imgs,
                max_outputs=1,
                step=step
            )

            # -> Write the ground truth
            disp_gts = data.get('GroundTruth')
            tf.summary.image(
                'GroundTruth',
                disp_gts,
                max_outputs=1,
                step=step
            )

            # -> Write the segmentations
            disp_segs = data.get('Segmentations')
            tf.summary.image(
                'Segmentations',
                disp_segs,
                max_outputs=1,
                step=step
            )

            # -> Write the scatter plot
            tf.summary.image(
                'Scatter',
                get_image_from_figure(
                    figure=plot_scatter(
                        x=data.get('Scatter')['x'],
                        y=data.get('Scatter')['y'],
                        save_file=None
                    )
                ),
                step=step
            )
