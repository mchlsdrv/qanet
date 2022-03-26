import os
import re
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
from models import cnn

from callbacks.visualisation_callbacks import (
    ProgressLogCallback
)
from configs.general_configs import (
    CROP_SIZE,

    TRAIN_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_SEG_DIR,

    TEST_DIR,
    TEST_IMAGE_DIR,
    TEST_SEG_DIR,

    INFERENCE_IMAGE_DIR,

    OUTPUT_DIR,

    EPSILON,
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
    MODEL_CHECKPOINT_SAVE_FREQ,
)

# from utils.visualisation_utils.plotting_funcs import (
from utils.plotting_funcs import (
    plot_scatter
)

from utils.image_utils.image_aux import (
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


def get_model(input_image_dims: tuple, checkpoint_dir: pathlib.Path = None, logger: logging.Logger = None):
    weights_loaded = False

    model = cnn.RibCage(input_image_dims=input_image_dims)

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


def choose_gpu(gpu_id: int = 0, logger: logging.Logger = None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_id > -1:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on: {physical_gpus}
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

    # FLAGS
    # a) General parameters
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--data_from_single_dir', default=False, action='store_true', help='If the data should be taken from a single directory, or collected from several directories')

    parser.add_argument('--reload_data', default=False, action='store_true', help=f'If to reload data from files and overwrite the temp data')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR, help='The path to the top directory where the images and the segmentations are stored')
    parser.add_argument('--train_image_dir', type=str, default=TRAIN_IMAGE_DIR, help='The path to the directory where the images are stored')
    parser.add_argument('--train_seg_dir', type=str, default=TRAIN_SEG_DIR, help='The path to the directory where the corresponding segmentations are stored')

    parser.add_argument('--test', default=False, action='store_true', help=f'If to perform the test of the current network')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help=f'Path to the test images, their segmentations and a file with the corresponding seg measures. Images should be placed in a folder called \'imgs\', the segmentations in a folder called \'segs\', and the file with the seg measures should be called \'seg_measures.pkl\', and be placed together with the two previous folders')
    parser.add_argument('--test_image_dir', type=str, default=TEST_IMAGE_DIR, help=f'Path to the test image directory')
    parser.add_argument('--test_seg_dir', type=str, default=TEST_SEG_DIR, help=f'Path to the test segmentations directory')

    parser.add_argument('--inference', default=False, action='store_true', help=f'If to perform the inference with the current network')
    parser.add_argument('--inference_image_dir', type=str, default=INFERENCE_IMAGE_DIR, help=f'Path to the images to infer directory')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    # b) Augmentations
    parser.add_argument('--crop_size', type=int, default=CROP_SIZE, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    # c) Network
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE, help='The number of samples in each validation batch')
    parser.add_argument('--validation_proportion', type=float, default=VALIDATION_PROPORTION, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the model')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--no_reduce_lr_on_plateau', default=False, action='store_true', help=f'If not to use the ReduceLROnPlateau callback')

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


def get_jaccard(gt_batch, seg_batch):
    """
    Receives two batches one of the predicted segmentations and the other of the
    ground truth segmentations, and returns the Jaccard measure i.e., the intersection
    over union (J = I / U)
    :gt_batch: The ground truth segmentation batch in format NHWC
    :seg_batch: The predicted segmentation batch in format NHWC
    :return:
        1) J - np.array of jaccard measures
        2) I - np.array of the intersection quantities
        3) U - np.array of the union quantities
    """
    # - Transform the gt_batch image into a binary format
    if np.max(gt_batch) > 1:
        gt_batch = np.array(gt_batch > 0, dtype=np.int16)

    # - Transform the seg_batch image into a binary format
    if np.max(seg_batch) > 1:
        seg_batch = np.array(seg_batch > 0, dtype=np.int16)

    # - Calculate the intersection of the images
    I = np.logical_and(gt_batch, seg_batch).sum(axis=(1, 2))

    # - Calculate the union of the images
    U = np.logical_or(gt_batch, seg_batch).sum(axis=(1, 2))

    # - Calculate the Jaccard coefficient
    J = I / (U + EPSILON)

    return J, I, U


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
