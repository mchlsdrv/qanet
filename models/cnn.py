import yaml
import logging
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from configs.general_configs import (
    DEBUG_LEVEL,
    PROFILE,
    RIBCAGE_CONFIGS_FILE_PATH
)

from utils.general_utils import aux_funcs
# from utils.general_utils.aux_funcs import (
#     get_runtime,
#     info_log,
# )


# def get_runtime(seconds: int):
#     hrs = int(seconds // 3600)
#     min = int((seconds - hrs * 3600) // 60)
#     sec = seconds - hrs * 3600 - min * 60
#
#     # - Format the strings
#     hrs_str = str(hrs)
#     if hrs_str < 10:
#         hrs_str = '0' + hrs_str
#     min_str = str(min)
#     if min_str < 10:
#         min_str = '0' + min_str
#     sec_str = str(min)
#     if sec < 10:
#         sec_str = '0' + sec_str
#
#     return f'{hrs_str}:{min_str}:{sec_str:.2f} [H:M:S]'
#
#
#
# def info_log(logger: logging.Logger, message: str):
#     if isinstance(logger, logging.Logger):
#         logger.info(message)
#     else:
#         print(message)


class RibCage(keras.Model):
    def __init__(self, input_image_dims: tuple, logger: logging.Logger = None):
        super().__init__()
        self.input_image_dims = input_image_dims
        self.logger = logger

        # - Open the models' configurations file
        self.ribcage_configs = None
        with RIBCAGE_CONFIGS_FILE_PATH.open(mode='r') as config_file:
            self.ribcage_configs = yaml.safe_load(config_file)

        # - Build the model
        self.model = self.build_model()

        # - Train epoch history
        self.train_imgs = None
        self.train_pred_segs = None
        self.train_epoch_trgt_seg_msrs = np.array([])
        self.train_epoch_pred_seg_msrs = np.array([])

        # - Validation epoch history
        self.val_imgs = None
        self.val_pred_segs = None
        self.val_epoch_trgt_seg_msrs = np.array([])
        self.val_epoch_pred_seg_msrs = np.array([])

    @staticmethod
    def _build_conv2d_block(filters: int, kernel_size: int):
        return keras.Sequential(
            [
                layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.MaxPool2D(padding='same')
            ]
        )

    @staticmethod
    def _build_fully_connected_block(units: int, drop_rate: float):
        return keras.Sequential(
            [
                keras.layers.Dense(units=units),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(),
                keras.layers.Dropout(rate=drop_rate)
            ]
        )

    def build_model(self):
        block_filters, block_kernel_sizes = self.ribcage_configs.get('conv2d_blocks')['block_filters'], self.ribcage_configs.get('conv2d_blocks')['block_kernel_sizes']

        input_left_rib = tmp_input_left_rib = keras.Input(self.input_image_dims + (1, ), name='input_left_rib')
        input_right_rib = tmp_input_right_rib = keras.Input(self.input_image_dims + (1, ), name='input_right_rib')
        input_spine = tmp_input_spine = keras.layers.Concatenate()([input_left_rib, input_right_rib])

        for filters, kernel_size in zip(block_filters, block_kernel_sizes):
            tmp_input_left_rib = self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(tmp_input_left_rib)
            tmp_input_right_rib = self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(tmp_input_right_rib)
            tmp_input_spine = keras.layers.Concatenate()(
                [
                    tmp_input_left_rib,
                    tmp_input_right_rib,
                    self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(tmp_input_spine)
                ]
            )

        layer_units, drop_rate = self.ribcage_configs.get('fully_connected_block')['layer_units'], self.ribcage_configs.get('fully_connected_block')['drop_rate']
        fc_layer = keras.layers.Flatten()(input_spine)
        for units in layer_units:
            fc_layer = self._build_fully_connected_block(units=units, drop_rate=drop_rate)(fc_layer)

        output_layer = keras.layers.Dense(units=1)(fc_layer)

        return keras.Model(inputs=[input_left_rib, input_right_rib], outputs=[output_layer])

    def call(self, inputs, training: bool = False):
        return self.model(inputs)

    def save(self, save_path: pathlib.Path):
        self.model.save(save_path)

    def summary(self):
        return self.model.summary()

    def train_step(self, data):
        t_strt = time.time()
        # - Get the data of the current epoch
        (imgs, aug_segs), trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            pred_seg_msrs = self.model([imgs, aug_segs], training=True)
            loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)

        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        # - Add images
        self.train_imgs = imgs
        self.train_pred_segs = aug_segs

        # - Add the target seg measures to epoch history
        self.train_epoch_trgt_seg_msrs = np.append(self.train_epoch_trgt_seg_msrs, trgt_seg_msrs.numpy())

        # - Add the modified seg measures to epoch history
        self.train_epoch_pred_seg_msrs = np.append(self.train_epoch_pred_seg_msrs, pred_seg_msrs.numpy())

        if PROFILE:
            aux_funcs.info_log(logger=self.logger, message=f'Training on batch of size {imgs.shape} took {aux_funcs.get_runtime(seconds=time.time() - t_strt)}')

        # - Return the mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        t_strt = time.time()
        # - Get the data of the current epoch
        (imgs, aug_segs), trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        pred_seg_msrs = self.model([imgs, aug_segs], training=True)
        loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)

        # - Add images
        self.val_imgs = imgs
        self.val_pred_segs = aug_segs

        # - Add the target seg measures to epoch history
        self.val_epoch_trgt_seg_msrs = np.append(self.val_epoch_trgt_seg_msrs, trgt_seg_msrs.numpy())

        # - Add the modified seg measures to epoch history
        self.val_epoch_pred_seg_msrs = np.append(self.val_epoch_pred_seg_msrs, pred_seg_msrs.numpy())

        if PROFILE:
            aux_funcs.info_log(logger=self.logger, message=f'Validating on batch of size {imgs.shape} took {aux_funcs.get_runtime(seconds=time.time() - t_strt)}')

        return {metric.name: metric.result() for metric in self.metrics}
