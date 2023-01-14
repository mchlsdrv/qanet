import os
from tqdm import tqdm
import logging
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .tf_activations import (
    Swish
)


class RibCage(keras.Model):
    def __init__(self, model_configs: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
        super().__init__()
        self.input_image_dims = model_configs.get('input_image_dims')
        self.logger = logger
        self.activation_layer = self._get_activation(configs=model_configs.get('activation'))
        self.kernel_regularizer = self._get_kernel_regularizer(configs=model_configs.get('kernel_regularizer'))
        self.output_dir = output_dir
        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(output_dir)
        assert isinstance(self.output_dir, pathlib.Path), f'The save_dir parameter must be of types str or pathlib.Path, but {type(output_dir)} was provided!'
        if not self.output_dir.is_dir():
            os.makedirs(self.output_dir)

        # - Open the models' configurations file
        self.architecture = model_configs.get('architecture')
        # with MODEL_CONFIGS_FILE.open(mode='r') as config_file:
        #     self.architecture = yaml.safe_load(config_file)

        # - Build the model
        self.model = self.build_model()

        # - Train epoch history
        self.train_epch_losses = np.array([])
        self.train_epch_gt_seg_msrs = np.array([])
        self.train_epch_pred_seg_msrs = np.array([])
        self.train_pearson_rs = np.array([])
        self.train_mses = np.array([])
        self.train_btch_smpl_dict = dict()

        # - Validation epoch history
        self.val_epch_losses = np.array([])
        self.val_epch_gt_seg_msrs = np.array([])
        self.val_epch_pred_seg_msrs = np.array([])
        self.val_pearson_rs = np.array([])
        self.val_mses = np.array([])
        self.val_btch_smpl_dict = dict()

        self.learning_rate = 0.0

    @staticmethod
    def _get_activation(configs: dict):
        activation = None
        if configs.get('type') == 'swish':
            activation = Swish()
        elif configs.get('type') == 'relu':
            activation = tf.keras.layers.ReLU(max_value=configs.get('max_value'), negative_slope=configs.get('negative_slope'), threshold=configs.get('threshold'))
        elif configs.get('type') == 'leaky_relu':
            activation = tf.keras.layers.LeakyReLU(alpha=configs.get('alpha'))
        return activation

    @staticmethod
    def _get_kernel_regularizer(configs: dict):
        kernel_regularizer = None
        if configs.get('type') == 'l1':
            kernel_regularizer = tf.keras.regularizers.L1(l1=configs.get('l1'))
        elif configs.get('type') == 'l2':
            kernel_regularizer = tf.keras.regularizers.L2(l2=configs.get('l2'))
        elif configs.get('type') == 'l1l2':
            kernel_regularizer = tf.keras.regularizers.L2(l1=configs.get('l1'), l2=configs.get('l2'))
        elif configs.get('type') == 'orthogonal':
            kernel_regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=configs.get('factor'), l2=configs.get('mode'))
        return kernel_regularizer

    # @staticmethod
    def _build_conv2d_block(self, filters: int, kernel_size: int):
        return keras.Sequential(
            [
                layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=self.kernel_regularizer),
                layers.BatchNormalization(),
                self.activation_layer,
                layers.MaxPool2D(padding='same'),
            ]
        )

    def _build_fully_connected_block(self, units: int, drop_rate: float):
        return keras.Sequential(
            [
                keras.layers.Dense(units=units, kernel_regularizer=self.kernel_regularizer),
                keras.layers.BatchNormalization(),
                self.activation_layer,
                keras.layers.Dropout(rate=drop_rate)
            ]
        )

    def build_model(self):
        block_filters, block_kernel_sizes = self.architecture.get('conv2d_blocks')['out_channels'], self.architecture.get('conv2d_blocks')['kernel_sizes']

        input_left_rib = tmp_input_left_rib = keras.Input(self.input_image_dims + (1, ), name='input_left_rib')
        input_right_rib = tmp_input_right_rib = keras.Input(self.input_image_dims + (1, ), name='input_right_rib')
        input_spine = keras.layers.Concatenate()([input_left_rib, input_right_rib])

        for filters, kernel_size in zip(block_filters, block_kernel_sizes):
            tmp_input_left_rib = self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(tmp_input_left_rib)
            tmp_input_right_rib = self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(tmp_input_right_rib)
            input_spine = keras.layers.Concatenate()(
                [
                    tmp_input_left_rib,
                    tmp_input_right_rib,
                    self._build_conv2d_block(filters=filters, kernel_size=kernel_size)(input_spine)
                ]
            )

        layer_units, drop_rate = self.architecture.get('fc_blocks')['out_features'], self.architecture.get('fc_blocks')['drop_rate']
        fc_layer = keras.layers.Flatten()(input_spine)
        for units in layer_units:
            fc_layer = self._build_fully_connected_block(units=units, drop_rate=drop_rate)(fc_layer)

        output_layer = keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(fc_layer)

        return keras.Model(inputs=[input_left_rib, input_right_rib], outputs=[output_layer])

    def call(self, inputs, training: bool = False):
        return self.model(inputs)

    def save(self, save_path: pathlib.Path):
        self.model.save(save_path)

    def summary(self):
        return self.model.summary()

    def _log(self, images, masks, true_seg_measures, pred_seg_measures, training: bool = True):
        with tf.device('CPU:0'):
            # --------------------------------------------------------------
            # - ADD THE HISTORY OF THE TRUE AND THE PREDICTED SEG MEASURES -
            # --------------------------------------------------------------
            if training:
                # - Add the target seg measures to epoch history
                self.train_epch_gt_seg_msrs = np.append(self.train_epch_gt_seg_msrs, true_seg_measures)

                # - Add the modified seg measures to epoch history
                self.train_epch_pred_seg_msrs = np.append(self.train_epch_pred_seg_msrs, pred_seg_measures)

                rnd_smpl_idx = np.random.randint(0, len(images) - 1)

                img = images[rnd_smpl_idx]
                msk = masks[rnd_smpl_idx]
                true_sm = true_seg_measures[rnd_smpl_idx]
                pred_sm = pred_seg_measures[rnd_smpl_idx]
                self.train_btch_smpl_dict = dict(image=img, mask=msk, true_seg_measure=true_sm, pred_seg_measure=pred_sm)

                self.learning_rate = self.optimizer.learning_rate.numpy()

            else:
                # - Add the target seg measures to epoch history
                self.val_epch_gt_seg_msrs = np.append(self.val_epch_gt_seg_msrs, true_seg_measures)

                # - Add the modified seg measures to epoch history
                self.val_epch_pred_seg_msrs = np.append(self.val_epch_pred_seg_msrs, pred_seg_measures)

                rnd_smpl_idx = np.random.randint(0, len(images) - 1)

                img = images[rnd_smpl_idx]
                msk = masks[rnd_smpl_idx]
                true_sm = true_seg_measures[rnd_smpl_idx]
                pred_sm = pred_seg_measures[rnd_smpl_idx]
                self.val_btch_smpl_dict = dict(image=img, mask=msk, true_seg_measure=true_sm, pred_seg_measure=pred_sm)

    def train_step(self, data) -> dict:
        # - Get the data of the current epoch
        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data

        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            btch_pred_seg_msrs = self.model([btch_imgs_aug, btch_msks_aug], training=True)
            loss = self.compiled_loss(btch_true_seg_msrs, btch_pred_seg_msrs)
        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Logs
        self._log(
            images=btch_imgs_aug.numpy(),
            masks=btch_msks_aug.numpy(),
            true_seg_measures=btch_true_seg_msrs.numpy(),
            pred_seg_measures=btch_pred_seg_msrs.numpy()[:, 0],
            training=True
        )
        self.train_epch_losses = np.append(self.train_epch_losses, loss.numpy())

        # - Return the mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data) -> dict:
        # - Get the data of the current epoch
        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data

        # - Compute the loss according to the predictions
        btch_pred_seg_msrs = self.model([btch_imgs_aug, btch_msks_aug], training=True)
        loss = self.compiled_loss(btch_true_seg_msrs, btch_pred_seg_msrs)

        self._log(
            images=btch_imgs_aug.numpy(),
            masks=btch_msks_aug.numpy(),
            true_seg_measures=btch_true_seg_msrs.numpy(),
            pred_seg_measures=btch_pred_seg_msrs.numpy()[:, 0],
            training=False
        )
        self.val_epch_losses = np.append(self.val_epch_losses, loss.numpy())

        return {metric.name: metric.result() for metric in self.metrics}

    def infer(self, data_loader) -> np.ndarray:
        t_strt = time.time()

        results = np.array([])

        # - Get the data of the current epoch
        for imgs, segs in tqdm(data_loader):
            # - Get the predictions
            pred_seg_msrs = self.model([imgs, segs], training=False)
            pred_seg_msrs = pred_seg_msrs.numpy()#[:, 0]

            # - Append the predicted seg measures to the results
            results = np.append(results, pred_seg_msrs)

        return results
