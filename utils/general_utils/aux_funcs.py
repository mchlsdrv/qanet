import os
import yaml
import logging
import logging.config
import threading
import argparse
import pathlib
import tensorflow as tf
import numpy as np
from models import cnn
from configs.general_configs import (
    IMAGES_DIR ,
    SEGMENTATIONS_DIR,
    OUTPUT_DIR,
    EPSILON,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_PROPORTION,
)


def decode_file(file):
    if isinstance(file, bytes):
        file = file.decode('utf-8')
    return file


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


def get_model(checkpoint_dir: pathlib.Path = None, logger: logging.Logger = None):
    weights_loaded = False

    model = cnn.RibCage()

    if checkpoint_dir.is_dir:
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)

            model.load_weights(latest_cpt)
            weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)
        else:
            if isinstance(logger, logging.Logger):
                logger.info(f'Weights from \'checkpoint_dir\' were loaded successfully to the \'RibCage\' model!')

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
        print(err)

    return logger


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # FLAGS
    # a) General parameters
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--images_dir', type=str, default=IMAGES_DIR, help='The path to the directory where the images are stored')
    parser.add_argument('--segmentations_dir', type=str, default=SEGMENTATIONS_DIR, help='The path to the directory where the corresponding segmentaions are stored')
    parser.add_argument('--test_data_dir', type=str, default='', help=f'Path to the test images, their segmentations and a file with the corresponding seg measures. Images should be placed in a folder called \'imgs\', the segmentations in a folder called \'segs\', and the file with the seg measures should be called \'seg_measures.pkl\', and be placed together with the two previous folders')
    parser.add_argument('--inference_data_dir', type=str, default='', help=f'Path to the images to infere, and their segmentations and a file with the corresponding seg measures. Images should be placed in a folder called \'imgs\', the segmentations in a folder called \'segs\'')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    # b) Augmentations
    parser.add_argument('--crop_size', type=int, default=CROP_SIZE, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    # c) Network
    parser.add_argument('--train_epochs', type=int, default=EPOCHS, help='Number of epochs to train the feature extractor network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the feature extraction model')
    parser.add_argument('--learning_rater', type=float, default=LEARNING_RATE, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--validation_proportion', type=float, default=VALIDATION_PROPORTION, help=f'The proportion of the data which will be set aside, and be used in the process of validation')

    # d) Flags
    parser.add_argument('--no_reduce_lr_on_plateau', default=False, action='store_true', help=f'If not to use the ReduceLROnPlateau callback')

    return parser


def get_file_names(images_dir: pathlib.Path, segmentations_dir: pathlib.Path):
    img_fls, seg_fls = list(), list()

    for root, dirs, files in os.walk(images_dir):
        for file in files:
            img_fls.append(f'{root}/{file}')

    for root, dirs, files in os.walk(segmentations_dir):
        for file in files:
            seg_fls.append(f'{root}/{file}')

    return list(zip(img_fls, seg_fls))


def get_train_val_split(data: np.ndarray, validation_proportion: float = .2):
    n_items = len(data)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * validation_proportion)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Pick the items for the validation set
    val_data = data[val_idxs]

    # - The items for training are the once which are not included in the validation set
    train_data = data[np.setdiff1d(item_idxs, val_idxs)]

    return train_data, val_data


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


    def train_model(model, data: dict, callback_configs: dict, compile_configs: dict, fit_configs: dict, general_configs: dict, logger: logging.Logger = None):

        # 2 - Train model
        # 2.2 Configure callbacks

        # 2.3 Compile model
        model.compile(
            loss=compile_configs.get('loss'),
            optimizer=compile_configs.get('optimizer'),
            metrics=compile_configs.get('metrics')
        )

        # 2.4 Fit model
        validation_steps = int(fit_configs.get('validation_steps_proportion') * fit_configs.get('train_steps_per_epoch')) if 0 < int(fit_configs.get('validation_steps_proportion') * fit_configs.get('train_steps_per_epoch')) <= fit_configs.get('train_steps_per_epoch') else 1
        model.fit(
            data.get('train_dataset'),
            batch_size=fit_configs.get('batch_size'),
            epochs=fit_configs.get('train_epochs'),
            steps_per_epoch=fit_configs.get('train_steps_per_epoch'),
            validation_data=data.get('val_dataset'),
            validation_steps=validation_steps,
            validation_freq=fit_configs.get('valdation_freq'),  # [1, 100, 1500, ...] - validate on these epochs
            shuffle=fit_configs.get('shuffle'),
            callbacks=callbacks
        )

        def test_step(self, data):
            D_btch, N_btch = data
            phi_X_btch = self.model(self.augmentations(D_btch), training=False)
            phi_ngbrs_btch = self.model(self.augmentations(N_btch), training=False)

            loss = self.compiled_loss(phi_X_btch, phi_ngbrs_btch, regularization_losses=self.losses)

           # Update the metrics
            self.compiled_metrics.update_state(phi_X_btch, phi_ngbrs_btch)

            # Return the mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}
        return model
