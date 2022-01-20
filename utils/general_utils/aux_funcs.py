import os
import yaml
import logging
import threading
import argparse
import pathlib
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models import cnn
from utils.image_utils import image_funcs
from configs.general_configs import (
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    OUTPUT_DIR,
    CONFIGS_DIR_PATH,
    EPSILON,
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
                logger.info(f'Weights from \'checkpoint_dir\' were loaded successfully to the \'{model_name}\' model!')

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


def get_arg_parcer():
    parser = argparse.ArgumentParser()

    # FLAGS
    # a) General parameters
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')
    parser.add_argument('--train_data_dir', type=str, default=TRAIN_DATA_DIR, help='The path to the train data directory')
    parser.add_argument('--test_data_dir', type=str, default=TEST_DATA_DIR, help='The path to the test data directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    # b) Augmentations
    parser.add_argument('--crop_size', type=int, default=256, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    # c) Network
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of epochs to train the feature extractor network')
    parser.add_argument('--train_steps_per_epoch', type=int, default=1000, help='Number of iterations that will be performed on each epoch for the featrue extractor network')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in each batch')
    parser.add_argument('--validation_split', type=float, default=0.1, help='The proportion of the data to be used for validation in the train process of the feature extractor model ((should be in range [0.0, 1.0])')
    parser.add_argument('--validation_steps_proportion', type=float, default=0.5, help='The proportion of validation steps in regards to the training steps in the train process of the feature extractor model ((should be in range [0.0, 1.0])')
    parser.add_argument('--checkpoint_dir', type=str, default='', help=f'The path to the directory that contains the checkpoints of the feature extraction model')
    parser.add_argument('--optimizer_lr', type=float, default=1e-4, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--no_reduce_lr_on_plateau', default=False, action='store_true', help=f'If not to use the ReduceLROnPlateau callback')
    parser.add_argument('--no_train', default=False, action='store_true', help=f'If theres no need to train the rib cage model')

    return parser


def get_jaccard(gt_batch, seg_batch):
    '''
    Receives two batches one of the predicted segmentations and the other of the
    ground truth segmentations, and returns the Jaccard measure i.e., the intersection
    over union (J = I / U)
    :gt_batch: The ground truth segmentation batch in format NHWC
    :seg_batch: The predicted segmentation batch in format NHWC
    :return:
        1) J - np.array of jaccard measurs
        2) I - np.array of the intersection quantities
        3) U - np.array of the union quantities
    '''
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

    # - Calculat the Jaccard coefficient
    J = I / (U + EPSILON)

    return J, I, U


def get_seg_measure(ground_truth_segmentations, predicted_segmentations):
    labels = np.where(np.unique(ground_truth_segmentations) > 0)[0]  # Exclude the background
    print('labels:', labels)
    number_of_classes = labels.shape[0]
    gt_areas = np.array([])
    intersect_areas = np.array([])
    obj_jaccards = np.array([])
    for gt_cls in labels:
        print('gt_cls:', gt_cls)
        # - Prepare the ground truth label
        gt_lbl = np.zeros_like(ground_truth_segmentations)
        gt_lbl_px = np.argwhere(ground_truth_segmentations == gt_cls)
        print('gt:', ground_truth_segmentations[ground_truth_segmentations==gt_cls])
        print('gt_lbl_px', gt_lbl_px)
        gt_x, gt_y = gt_lbl_px[:, 0], gt_lbl_px[:, 1]
        gt_lbl[(gt_x, gt_y)] = 1
        gt_areas = np.append(gt_areas, gt_lbl.sum())

        # - Prepare the ground truth label
        # (*) As the predicted segmentations' class may differ, we need to infere it
        # from the ground truth, by looking at the most abundant label in the vacinity
        # of the ground truth label
        I = gt_lbl * predicted_segmentations  # => provides the labels which correspond to the ground truth label
        I = I[I > 0]  # => as '0' will be there, we need to dispose of it, as it is the background
        seg_lbls, cnts = np.unique(I, return_counts=True)  # => counting the labels
        seg_cls = seg_lbls[np.argmax(cnts)]  # => here we choose the most abundant label as the class we are looking for in the predictions

        seg_lbl = np.zeros_like(predicted_segmentations)
        seg_lbl_px = np.argwhere(predicted_segmentations == seg_cls)
        print('seg:', predicted_segmentations[predicted_segmentations==seg_cls])
        seg_x, seg_y = seg_lbl_px[:, 0], seg_lbl_px[:, 1]
        seg_lbl[(seg_x, seg_y)] = 1

        # - Calculate the intersection of the ground truth segmentation with
        # the predicted segmentation
        I_area = np.logical_and(gt_lbl, seg_lbl).sum()
        print('I:', I_area)
        intersect_areas = np.append(intersect_areas, I_area)

        # - Calculate the union of the ground truth segmentation with
        # the predicted segmentation
        U_area = np.logical_or(gt_lbl, seg_lbl).sum()
        print('U:', U_area)

        # - Calculate the Jaccard coefficient of the current label
        J = I_area / (U_area + EPSILON)
        obj_jaccards = np.append(obj_jaccards, J)
    print('obj_jacc:', obj_jaccards)
    print('gt_areas:', gt_areas)
    # - The detection counts only if the intersection is greater then INTERSECTION_TH
    dets = intersect_areas / (gt_areas + EPSILON) > DETECTION_TH
    print('dets:', dets)
    # - The seg measure is the sum of the jaccards of the detected objects to
    # the number of different objects in the image
    seg_measure = obj_jaccards[dets].sum() / (number_of_classes + EPSILON)

    return seg_measure

# def get_seg_measure(gt_batch, seg_batch):
#     btch_seg_measures = np.array([])
#     for idx, (gt, seg) in enumerate(zip(gt_batch, seg_batch)):
#         labels = np.where(np.unique(gt) > 0)  # Exclude the background
#         number_of_classes = labels.shape[0]
#         gt_areas = np.array([])
#         intersect_areas = np.array([])
#         jaccards = np.array([])
#         for lbl in labels:
#             # - Prepare the ground truth label
#             gt_lbl = np.zeros_like(gt)
#             gt_lbl_px = np.argwhere(gt_lbl == lbl)
#             gt_x, gt_y = gt_lbl_px[:, 0], gt_lbl_px[:, 1]
#             gt_lbl[(gt_x, gt_y)] = 1
#             gt_areas = np.append(gt_areas, gt_lbl.sum())
#
#             # - Prepare the ground truth label
#             seg_lbl = np.zeros_like(seg)
#             seg_lbl_px = np.argwhere(seg_lbl == lbl)
#             seg_x, seg_y = seg_lbl_px[:, 0], seg_lbl_px[:, 1]
#             seg_lbl[(seg_x, seg_y)] = 1
#
#             # - Calculate the intersection of the ground truth segmentation with
#             # the predicted segmentation
#             I = np.logical_and(gt_lbl, seg_lbl)
#             intersect_areas = np.append(intersect_areas, I.sum())
#
#             # - Calculate the union of the ground truth segmentation with
#             # the predicted segmentation
#             U = np.logical_or(gt_lbl, seg_lbl)
#
#             # - Calculate the Jaccard coefficient of the current label
#             J = I / (U + EPSILON)
#             jaccards = np.append(jaccards, J)
#
#         # - The detection counts only if the intersection is greater then INTERSECTION_TH
#         dets = intersect_areas / (gt_areas + EPSILON) > DETECTION_TH
#
#         # - The seg measure is the sum of the jaccards of the detected objects to
#         # the number of different objects in the image
#         seg_measure = jaccards[dets].sum() / (number_of_classes + EPSILON)
#
#         btch_seg_measures = np.append(btch_seg_measures, seg_measure)
#
#     return seg_measure
#
#
#
