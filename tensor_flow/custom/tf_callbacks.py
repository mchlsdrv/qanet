import os
import pathlib
import tensorflow as tf
import numpy as np
import logging
from utils.aux_funcs import (
    scatter_plot,
    absolute_error_plot, hit_rate_plot,
)


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: pathlib.Path or str, logger: logging.Logger = None):
        super().__init__()
        self.log_dir = log_dir
        assert isinstance(self.log_dir, pathlib.Path) or isinstance(self.log_dir, str), f'The parameter \'log_dir\' must be of types \'pathlib.Path\', or \'str\', but is of type \'{type(log_dir)}\'!'
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        # Train
        # - Scatter plot
        scatter_plot(
            x=self.model.train_true_seg_msrs,
            y=self.model.train_pred_seg_msrs,
            save_file=self.log_dir / f'train/plots/scatter_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Absolute error plot
        absolute_error_plot(
            true=self.model.train_true_seg_msrs,
            pred=self.model.train_pred_seg_msrs,
            save_file=self.log_dir / f'train/plots/abs_err_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Hit rate plot
        hit_rate_plot(
            true=self.model.train_true_seg_msrs,
            pred=self.model.train_pred_seg_msrs,
            save_file=self.log_dir / f'train/plots/hit_rate_plots/step_{epoch}.png',
            logger=self.logger
        )

        # Validation
        # - Scatter plot
        scatter_plot(
            x=self.model.val_true_seg_msrs,
            y=self.model.val_pred_seg_msrs,
            save_file=self.log_dir / f'validation/plots/scatter_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Absolute error plot
        absolute_error_plot(
            true=self.model.val_true_seg_msrs,
            pred=self.model.val_pred_seg_msrs,
            save_file=self.log_dir / f'validation/plots/abs_err_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Hit rate plot
        hit_rate_plot(
            true=self.model.val_true_seg_msrs,
            pred=self.model.val_pred_seg_msrs,
            save_file=self.log_dir / f'validation/plots/hit_rate_plots/step_{epoch}.png',
            logger=self.logger
        )
        # Clean the data for the next epoch
        self.model.train_true_seg_msrs = np.array([])
        self.model.train_pred_seg_msrs = np.array([])
        self.model.val_true_seg_msrs = np.array([])
        self.model.val_pred_seg_msrs = np.array([])
