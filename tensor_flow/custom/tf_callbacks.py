import os
import pathlib
import tensorflow as tf
import numpy as np
import logging
from utils.aux_funcs import (
    scatter_plot,
    absolute_error_plot, hit_rate_plot, to_numpy, err_log,
)


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: pathlib.Path or str, logger: logging.Logger = None):
        super().__init__()

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Train
            # - Scatter plot
            train_scatter_plots_dir = self.log_dir / f'train/plots/scatter'
            scatter_plot(
                x=np.array(self.model.train_gt_seg_msrs).flatten(),
                y=np.array(self.model.train_pred_seg_msrs).flatten(),
                save_file=train_scatter_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=np.array(self.model.train_gt_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.train_pred_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Absolute error plot
            train_abs_err_plots_dir = self.log_dir / f'train/plots/absolute_error'
            train_abs_err_hist, train_abs_err_bins = absolute_error_plot(
                true=np.array(self.model.train_gt_seg_msrs).flatten(),
                pred=np.array(self.model.train_pred_seg_msrs).flatten(),
                save_file=train_abs_err_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=train_abs_err_hist, file_path=train_abs_err_plots_dir / f'metadata/abs_error_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=train_abs_err_bins, file_path=train_abs_err_plots_dir / f'metadata/abs_error_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Hit rate plot
            train_hit_rate_plots_dir = self.log_dir / f'train/plots/hit_rate'
            train_hit_rate_hist, train_hit_rate_bins = hit_rate_plot(
                true=np.array(self.model.train_gt_seg_msrs).flatten(),
                pred=np.array(self.model.train_pred_seg_msrs).flatten(),
                save_file=train_hit_rate_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=train_hit_rate_hist, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=train_hit_rate_bins, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # Validation
            # - Scatter plot
            val_scatter_plots_dir = self.log_dir / f'validation/plots/scatter'
            scatter_plot(
                x=np.array(self.model.val_gt_seg_msrs).flatten(),
                y=np.array(self.model.val_pred_seg_msrs).flatten(),
                save_file=val_scatter_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=np.array(self.model.val_gt_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.val_pred_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Absolute error plot
            val_abs_err_plots_dir = self.log_dir / f'validation/plots/absolute_error'
            val_abs_err_hist, val_abs_err_bins = absolute_error_plot(
                true=np.array(self.model.val_gt_seg_msrs).flatten(),
                pred=np.array(self.model.val_pred_seg_msrs).flatten(),
                save_file=val_abs_err_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=val_abs_err_hist, file_path=val_abs_err_plots_dir / f'metadata/abs_error_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=val_abs_err_bins, file_path=val_abs_err_plots_dir / f'metadata/abs_error_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Hit rate plot
            val_hit_rate_plots_dir = self.log_dir / f'validation/plots/hit_rate'
            val_hit_rate_hist, val_hit_rate_bins = hit_rate_plot(
                true=np.array(self.model.val_gt_seg_msrs).flatten(),
                pred=np.array(self.model.val_pred_seg_msrs).flatten(),
                save_file=val_hit_rate_plots_dir / f'step_{epoch}.png',
                logger=self.logger
            )
            to_numpy(data=val_hit_rate_hist, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=val_hit_rate_bins, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

        except RuntimeError as err:
            err_log(logger=self.logger, message=f'{err}')

        # - Clean the data for the next epoch
        self.model.train_gt_seg_msrs = []
        self.model.train_pred_seg_msrs = []

        self.model.val_gt_seg_msrs = []
        self.model.val_pred_seg_msrs = []
