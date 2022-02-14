import yaml
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from configs.general_configs import (
    RIBCAGE_CONFIGS_FILE_PATH
)


class RibCage(keras.Model):
    def __init__(self):
        super().__init__()

        # - Open the models' configurations file
        self.ribcage_configs = None
        with RIBCAGE_CONFIGS_FILE_PATH.open(mode='r') as config_file:
            self.ribcage_configs = yaml.safe_load(config_file)

        # - Build the model
        self.model = self.build()

        # - Metrics
        self.train_loss = tf.metrics.Mean(name='train_loss')
        self.val_loss = tf.metrics.Mean(name='val_loss')

        # - Train epoch history
        self.train_epoch_trgt_seg_msrs = np.array([])
        self.train_epoch_pred_seg_msrs = np.array([])

        # - Validation epoch history
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

    def build(self):
        image_dims = self.ribcage_configs.get('image_dims')
        block_filters, block_kernel_sizes = self.ribcage_configs.get('conv2d_blocks')['block_filters'], self.ribcage_configs.get('conv2d_blocks')['block_kernel_sizes']

        input_left_rib = tmp_input_left_rib = keras.Input(image_dims + (3, ), name='input_left_rib')
        input_right_rib = tmp_input_right_rib = keras.Input(image_dims + (1, ), name='input_right_rib')
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

    def train_step(self, data):

        # - Get the data of the current epoch
        imgs, mod_segs, trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            pred_seg_msrs = self.model([imgs, mod_segs], training=True)
            loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)

        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # - Update the metrics
        self.train_loss(loss)

        # - Add the target seg measures to epoch history
        self.train_epoch_trgt_seg_msrs = np.append(self.train_epoch_trgt_seg_msrs, trgt_seg_msrs)

        # - Add the modified seg measures to epoch history
        self.train_epoch_pred_seg_msrs = np.append(self.train_epoch_pred_seg_msrs, pred_seg_msrs)

        # - Return the mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        # - Get the data of the current epoch
        imgs, mod_segs, trgt_seg_msrs = data

        # - Compute the loss according to the predictions
        pred_seg_msrs = self.model([imgs, mod_segs], training=True)
        loss = self.compiled_loss(trgt_seg_msrs, pred_seg_msrs)
        self.val_loss(loss)

        # - Add the target seg measures to epoch history
        self.val_epoch_trgt_seg_msrs = np.append(self.val_epoch_trgt_seg_msrs, trgt_seg_msrs)

        # - Add the modified seg measures to epoch history
        self.val_epoch_pred_seg_msrs = np.append(self.val_epoch_pred_seg_msrs, pred_seg_msrs)

        return {metric.name: metric.result() for metric in self.metrics}

    def unit_test(self):
        input_shape_left = (160, 160, 3)
        input_shape_right = (160, 160, 1)
        iterations = 10
        for _ in range(iterations):
            left = np.random.randn(5, *input_shape_left).astype(np.float32)
            right = np.random.randn(5, *input_shape_right).astype(np.float32)
            output = self.model([left, right], training=True)
            print(output)


if __name__ == '__main__':
    mdl = RibCage()
    mdl.unit_test()
