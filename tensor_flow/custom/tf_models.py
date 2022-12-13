import os
import yaml
from tqdm import tqdm
import logging
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from global_configs.general_configs import (
    MODEL_CONFIGS_FILE,
)
from utils.aux_funcs import plot_image_mask, float_2_str

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
        self.ribcage_configs = None
        with MODEL_CONFIGS_FILE.open(mode='r') as config_file:
            self.ribcage_configs = yaml.safe_load(config_file)

        # - Build the model
        self.model = self.build_model()

        # - Train epoch history
        self.train_epch_gt_seg_msrs = np.array([])
        self.train_epch_pred_seg_msrs = np.array([])
        self.train_pearson_rs = np.array([])
        self.train_mses = np.array([])
        self.train_btch_smpl_dict = dict()

        # - Validation epoch history
        self.val_epch_gt_seg_msrs = np.array([])
        self.val_epch_pred_seg_msrs = np.array([])
        self.val_pearson_rs = np.array([])
        self.val_mses = np.array([])
        self.val_btch_smpl_dict = dict()

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

                # TODO: Instead of figure save a sample and plot in the callback
                rnd_smpl_idx = np.random.randint(0, len(images) - 1)

                img = images[rnd_smpl_idx]
                msk = masks[rnd_smpl_idx]
                true_sm = true_seg_measures[rnd_smpl_idx]
                pred_sm = pred_seg_measures[rnd_smpl_idx]
                self.train_btch_smpl_dict = dict(image=img, mask=msk, true_seg_measure=true_sm, pred_seg_measure=pred_sm)

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

    # def train(self, epochs, train_data_loader, val_data_loader):
    #
    #     train_losses = np.array([])
    #     val_losses = np.array([])
    #     best_val_loss = np.inf
    #     for epch in range(epochs):
    #         print(f'\n > Starting epoch #{epch+1}')
    #
    #         # - TRAIN
    #         # - Iterate over the batches of the data loader
    #         data_tuples_pbar = tqdm(train_data_loader)
    #         step = 0
    #         for btch_imgs, btch_gt_msks, btch_aug_msks, btch_gt_seg_msrs in data_tuples_pbar:
    #             train_loss, btch_pred_seg_msrs = self.train_step(btch_imgs, btch_aug_msks, btch_gt_seg_msrs)
    #
    #             # if step % 100 == 0:
    #             #     print(f'Train loss for step {step}: {float(train_loss):.4f}')
    #
    #             btch_train_losses = np.array([])
    #             with tf.device('CPU:0'):
    #                 btch_train_losses = np.append(btch_train_losses, train_loss)
    #                 data_tuples_pbar.set_postfix({'train_loss': np.round(train_loss, decimals=4)})
    #                 # --------------------------------------------------------------
    #                 # - ADD THE HISTORY OF THE TRUE AND THE PREDICTED SEG MEASURES -
    #                 # --------------------------------------------------------------
    #                 # - Add the target seg measures to epoch history
    #                 self.train_epch_gt_seg_msrs.append(btch_gt_seg_msrs)
    #
    #                 # - Add the modified seg measures to epoch history
    #                 self.train_epch_pred_seg_msrs.append(btch_pred_seg_msrs)
    #
    #                 # - Add the ground truth segmentations batch
    #                 self.train_gt_msks.append(btch_gt_msks)
    #
    #                 # - Add the predicted segmentations batch
    #                 self.train_aug_msks.append(btch_aug_msks)
    #             step += 1
    #
    #         with tf.device('CPU:0'):
    #             train_losses = np.append(train_losses, btch_train_losses.mean())
    #             # - Scatter plot
    #             scatter_plot(
    #                 x=np.array(self.train_epch_gt_seg_msrs).flatten(),
    #                 y=np.array(self.train_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'train/plots/scatter_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #             # - Absolute error plot
    #             absolute_error_plot(
    #                 true=np.array(self.train_epch_gt_seg_msrs).flatten(),
    #                 pred=np.array(self.train_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'train/plots/abs_err_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #             # - Hit rate plot
    #             hit_rate_plot(
    #                 true=np.array(self.train_epch_gt_seg_msrs).flatten(),
    #                 pred=np.array(self.train_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'train/plots/hit_rate_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #
    #             train_rnd_idx = np.random.randint(0, len(self.train_epch_gt_seg_msrs)-1)
    #             monitor_seg_error(
    #                 ground_truth=self.train_gt_msks[train_rnd_idx],
    #                 prediction=self.train_aug_msks[train_rnd_idx],
    #                 seg_measures=self.train_epch_gt_seg_msrs[train_rnd_idx],
    #                 save_dir=self.output_dir / f'train/plots/error_monitor/epoch_{epch}'
    #             )
    #
    #         # - VALIDATION
    #         data_tuples_pbar = tqdm(val_data_loader)
    #         step = 0
    #         btch_val_losses = np.array([])
    #         for btch_imgs, btch_gt_msks, btch_aug_msks, btch_gt_seg_msrs in data_tuples_pbar:
    #             val_loss, btch_pred_seg_msrs = self.test_step(btch_imgs, btch_aug_msks, btch_gt_seg_msrs)
    #
    #             with tf.device('CPU:0'):
    #                 btch_val_losses = np.append(btch_val_losses, val_loss)
    #                 data_tuples_pbar.set_postfix({'val_loss': np.round(val_loss, decimals=4)})
    #                 # if step % 100 == 0:
    #                 #     print(f'Val loss for step {step}: {float(val_loss):.4f}')
    #                 # - Return the mapping metric names to current value
    #                 # --------------------------------------------------------------
    #                 # - ADD THE HISTORY OF THE TRUE AND THE PREDICTED SEG MEASURES -
    #                 # --------------------------------------------------------------
    #                 # - Add the target seg measures to epoch history
    #                 self.val_epch_gt_seg_msrs.append(btch_gt_seg_msrs)
    #
    #                 # - Add the modified seg measures to epoch history
    #                 self.val_epch_pred_seg_msrs.append(btch_pred_seg_msrs)
    #
    #                 # - Add the ground truth segmentations batch
    #                 self.val_gt_msks.append(btch_gt_msks)
    #
    #                 # - Add the predicted segmentations batch
    #                 self.val_aug_msks.append(btch_aug_msks)
    #
    #             step += 1
    #         with tf.device('CPU:0'):
    #             val_losses = np.append(val_losses, btch_val_losses.mean())
    #             # - Scatter plot
    #             scatter_plot(
    #                 x=np.array(self.val_epch_gt_seg_msrs).flatten(),
    #                 y=np.array(self.val_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'validation/plots/scatter_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #             # - Absolute error plot
    #             absolute_error_plot(
    #                 true=np.array(self.val_epch_gt_seg_msrs).flatten(),
    #                 pred=np.array(self.val_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'validation/plots/abs_err_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #             # - Hit rate plot
    #             hit_rate_plot(
    #                 true=np.array(self.val_epch_gt_seg_msrs).flatten(),
    #                 pred=np.array(self.val_epch_pred_seg_msrs).flatten(),
    #                 save_file=self.output_dir / f'validation/plots/hit_rate_plots/step_{epch}.png',
    #                 logger=self.logger
    #             )
    #             val_rnd_idx = np.random.randint(0, len(self.val_epch_gt_seg_msrs)-1)
    #             monitor_seg_error(
    #                 ground_truth=self.val_gt_msks[val_rnd_idx],
    #                 prediction=self.val_aug_msks[val_rnd_idx],
    #                 seg_measures=self.val_epch_gt_seg_msrs[val_rnd_idx],
    #                 save_dir=self.output_dir / f'validation/plots/error_monitor/epoch_{epch}'
    #             )
    #
    #             if val_loss < best_val_loss:
    #                 chpt_dir = self.output_dir / 'checkpoints'
    #                 os.makedirs(chpt_dir, exist_ok=True)
    #
    #                 print(f'''
    #                 - Loss improved from {best_val_loss:.4f} to {val_loss:.4f}!
    #                     > Saving checkpoint to {chpt_dir / 'best_val_loss'}
    #                 ''')
    #                 # self.save(chpt_dir / CHECKPOINT_FILE_BEST_MODEL_FILE_NAME)
    #                 self.save_weights(chpt_dir / 'beast_val_loss')
    #                 best_val_loss = val_loss
    #
    #         line_plot(
    #             x=np.arange(len(train_losses)),
    #             ys=[train_losses, val_losses],
    #             suptitle=f'Train Validation Loss Plot Epoch {epch}',
    #             labels=['train', 'val'],
    #             colors=('g', 'r', 'b'),
    #             save_file=self.output_dir / 'loss_plot.png'
    #         )
    #
    #         print(f'''
    #         ==============================================================
    #         Epoch #{epch+1} Stats:
    #             - Train loss: {btch_train_losses.mean():.4f}+/-{btch_train_losses.std():.4f}
    #             - Val loss: {btch_val_losses.mean():.4f}+/-{btch_val_losses.std():.4f}
    #         ==============================================================
    #         ''')
    #
    #         # Clean the data for the next epoch
    #         self.model.train_gt_seg_msrs = []
    #         self.model.train_epch_pred_seg_msrs = []
    #         self.model.train_gt_msks = []
    #         self.model.train_aug_msks = []
    #
    #         self.model.val_gt_seg_msrs = []
    #         self.model.val_pred_seg_msrs = []
    #         self.model.val_gt_msks = []
    #         self.model.val_aug_msks = []
    #
