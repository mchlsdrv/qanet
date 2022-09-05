import yaml
from tqdm import tqdm
import logging
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from configs.general_configs import (
    MODEL_CONFIGS_FILE,
    OUTLIER_TH,
)

from utils.aux_funcs import (
    scatter_plot
)
from .tf_activations import (
    Swish
)
from .tf_callbacks import (
    log_masks,
)


class RibCage(keras.Model):
    def __init__(self, model_configs: dict, logger: logging.Logger = None):
        super().__init__()
        self.input_image_dims = model_configs.get('input_image_dims')
        self.logger = logger
        self.activation_layer = self._get_activation(configs=model_configs.get('activation'))
        self.kernel_regularizer = self._get_kernel_regularizer(configs=model_configs.get('kernel_regularizer'))

        # - Open the models' configurations file
        self.ribcage_configs = None
        with MODEL_CONFIGS_FILE.open(mode='r') as config_file:
            self.ribcage_configs = yaml.safe_load(config_file)

        # - Build the model
        self.model = self.build_model()

        # - Train epoch history
        self.train_losses = []
        self.train_loss_prev = 1.
        self.train_loss_delta = 0.
        self.train_imgs = None
        self.train_aug_segs = None
        self.train_trgt_seg_msrs = None
        self.train_pred_seg_msrs = None
        self.train_epoch_trgt_seg_msrs = np.array([])
        self.train_epoch_pred_seg_msrs = np.array([])
        self.train_epoch_outliers = list()

        # - Validation epoch history
        self.val_losses = []
        self.val_loss_prev = 1.
        self.val_loss_delta = 0.
        self.val_imgs = None
        self.val_aug_segs = None
        self.val_trgt_seg_msrs = None
        self.val_pred_seg_msrs = None
        self.val_epoch_trgt_seg_msrs = np.array([])
        self.val_epoch_pred_seg_msrs = np.array([])
        self.val_epoch_outliers = list()

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
                layers.MaxPool2D(padding='same')
            ]
        )

    # @staticmethod
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
        block_filters, block_kernel_sizes = self.ribcage_configs.get('conv2d_blocks')['out_channels'], self.ribcage_configs.get('conv2d_blocks')['kernel_sizes']

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

        layer_units, drop_rate = self.ribcage_configs.get('fc_blocks')['out_features'], self.ribcage_configs.get('fc_blocks')['drop_rate']
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

    def train_step(self, data) -> dict:
        t_strt = time.time()
        # - Get the data of the current epoch
        (imgs, aug_segs), trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            pred_seg_msrs = self.model([imgs, aug_segs], training=True)
            loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)

        trgt_seg_msrs = trgt_seg_msrs.numpy()
        pred_seg_msrs = pred_seg_msrs.numpy()[:, 0]
        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        self.train_loss_delta = loss - self.train_loss_prev
        self.train_loss_prev = loss

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # - Add images
        self.train_imgs = imgs.numpy()
        self.train_aug_segs = aug_segs.numpy()
        self.train_trgt_seg_msrs = trgt_seg_msrs
        self.train_pred_seg_msrs = pred_seg_msrs

        train_segs = []
        for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.train_imgs, self.train_aug_segs, self.train_trgt_seg_msrs, self.train_pred_seg_msrs)):
            train_segs.append(log_masks(image=img, mask=seg, true_seg_measure=trgt_seg_msr, pred_seg_measure=pred_seg_msr))
            if idx == 5:
                break
            # train_segs = log_bboxes(image=self.train_imgs[0], mask=self.train_aug_segs[0], true_seg_measure=self.train_trgt_seg_msrs[0], pred_seg_measure=self.train_pred_seg_msrs[0], procedure='train')

        train_scatter_plot = scatter_plot(
            x=trgt_seg_msrs,
            y=pred_seg_msrs,
        )

        # - Add the target seg measures to epoch history
        self.train_epoch_trgt_seg_msrs = np.append(self.train_epoch_trgt_seg_msrs, trgt_seg_msrs)

        # - Add the modified seg measures to epoch history
        self.train_epoch_pred_seg_msrs = np.append(self.train_epoch_pred_seg_msrs, pred_seg_msrs)

        # - Add the outliers, if there's any
        seg_msrs_diff = np.abs(trgt_seg_msrs - pred_seg_msrs)
        outliers_idxs = np.argwhere(seg_msrs_diff > OUTLIER_TH).flatten()

        train_outlier_segs = []
        for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.train_imgs[outliers_idxs], self.train_aug_segs[outliers_idxs], trgt_seg_msrs[outliers_idxs], pred_seg_msrs[outliers_idxs])):
            train_outlier_segs.append(log_masks(image=img, mask=seg, true_seg_measure=trgt_seg_msr, pred_seg_measure=pred_seg_msr))

        # - Return the mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data) -> dict:
        # - Get the data of the current epoch
        (imgs, aug_segs), trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        pred_seg_msrs = self.model([imgs, aug_segs], training=True)
        loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)

        self.val_loss_delta = loss - self.val_loss_prev
        self.val_loss_prev = loss

        trgt_seg_msrs = trgt_seg_msrs.numpy()
        pred_seg_msrs = pred_seg_msrs.numpy()[:, 0]

        # - Update images
        self.val_imgs = imgs.numpy()
        self.val_aug_segs = aug_segs.numpy()
        self.val_trgt_seg_msrs = trgt_seg_msrs
        self.val_pred_seg_msrs = pred_seg_msrs
        # val_segs = log_bboxes(image=self.val_imgs[0], mask=self.val_aug_segs[0], true_seg_measure=self.val_trgt_seg_msrs[0], pred_seg_measure=self.val_pred_seg_msrs[0], procedure='validation')
        val_segs = []
        for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.val_imgs, self.val_aug_segs, self.val_trgt_seg_msrs, self.val_pred_seg_msrs)):
            val_segs.append(log_masks(image=img, mask=seg, true_seg_measure=trgt_seg_msr, pred_seg_measure=pred_seg_msr))
            if idx == 5:
                break
        val_scatter_plot = scatter_plot(
            x=trgt_seg_msrs,
            y=pred_seg_msrs,
        )
        # - Add the target seg measures to epoch history
        self.val_epoch_trgt_seg_msrs = np.append(self.val_epoch_trgt_seg_msrs, trgt_seg_msrs)

        # - Add the modified seg measures to epoch history
        self.val_epoch_pred_seg_msrs = np.append(self.val_epoch_pred_seg_msrs, pred_seg_msrs)

        # - Add the outliers, if there's any
        seg_msrs_diff = np.abs(trgt_seg_msrs - pred_seg_msrs)
        outliers_idxs = np.argwhere(seg_msrs_diff > OUTLIER_TH).flatten()

        val_outlier_segs = []
        for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.val_imgs[outliers_idxs], self.val_aug_segs[outliers_idxs], trgt_seg_msrs[outliers_idxs], pred_seg_msrs[outliers_idxs])):
            val_outlier_segs.append(log_masks(image=img, mask=seg, true_seg_measure=trgt_seg_msr, pred_seg_measure=pred_seg_msr))

        return {metric.name: metric.result() for metric in self.metrics}

    def infer(self, data_loader) -> np.ndarray:
        t_strt = time.time()

        results = np.array([])

        # - Get the data of the current epoch
        for imgs, segs, _ in tqdm(data_loader):
            # - Get the predictions
            pred_seg_msrs = self.model([imgs, segs], training=False)
            pred_seg_msrs = pred_seg_msrs.numpy()[:, 0]

            # - Append the predicted seg measures to the results
            results = np.append(results, pred_seg_msrs)

        return results
