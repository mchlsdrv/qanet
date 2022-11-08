import os
import pathlib
import tensorflow as tf
import numpy as np
import logging
from utils.aux_funcs import (
    scatter_plot,
    absolute_error_plot, hit_rate_plot, monitor_seg_error,
)


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: pathlib.Path or str, logger: logging.Logger = None):
        super().__init__()
        self.log_dir = log_dir
        assert isinstance(self.log_dir, pathlib.Path) or isinstance(self.log_dir, str), f'The parameter \'log_dir\' must be of types \'pathlib.Path\', or \'str\', but is of type \'{type(log_dir)}\'!'
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logger

    def on_epoch_start(self, epoch, logs=None):

        # Clean the data for the next epoch
        self.model.train_true_seg_msrs = []
        self.model.train_pred_seg_msrs = []
        self.model.train_gt_msks = []
        self.model.train_aug_msks = []

        self.model.val_true_seg_msrs = []
        self.model.val_pred_seg_msrs = []
        self.model.val_gt_msks = []
        self.model.val_aug_msks = []

    def on_epoch_end(self, epoch, logs=None):

        # btch_true_seg_msrs = btch_true_seg_msrs.numpy()
        # btch_pred_seg_msrs = btch_pred_seg_msrs.numpy()[:, 0]
        # monitor_data(image=btch_imgs_aug.numpy()[rnd_idx],  ground_truth=None, segmentation=btch_msks_aug.numpy()[rnd_idx], image_label='Image', ground_truth_label='Ground Truth', segmentation_label=f'Prediction (gt={btch_true_seg_msrs[rnd_idx]:.3f}, pred={btch_pred_seg_msrs[rnd_idx]:.3f})', save_file=str(train_imgs_dir / f'train_btch_{ts}.png'))
        # Train
        # - Scatter plot
        scatter_plot(
            x=np.array(self.model.train_true_seg_msrs).flatten(),
            y=np.array(self.model.train_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'train/plots/scatter_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Absolute error plot
        absolute_error_plot(
            true=np.array(self.model.train_true_seg_msrs).flatten(),
            pred=np.array(self.model.train_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'train/plots/abs_err_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Hit rate plot
        hit_rate_plot(
            true=np.array(self.model.train_true_seg_msrs).flatten(),
            pred=np.array(self.model.train_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'train/plots/hit_rate_plots/step_{epoch}.png',
            logger=self.logger
        )

        train_rnd_idx = np.random.randint(0, len(self.model.train_true_seg_msrs)-1)
        monitor_seg_error(
            ground_truth=self.model.train_gt_msks[train_rnd_idx],
            prediction=self.model.train_aug_msks[train_rnd_idx],
            seg_measures=self.model.train_gt_seg_msrs[train_rnd_idx],
            save_dir=f'train/plots/error_monitor/epoch_{epoch}'
        )

        # Validation
        # - Scatter plot
        scatter_plot(
            x=np.array(self.model.val_true_seg_msrs).flatten(),
            y=np.array(self.model.val_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'validation/plots/scatter_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Absolute error plot
        absolute_error_plot(
            true=np.array(self.model.val_true_seg_msrs).flatten(),
            pred=np.array(self.model.val_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'validation/plots/abs_err_plots/step_{epoch}.png',
            logger=self.logger
        )
        # - Hit rate plot
        hit_rate_plot(
            true=np.array(self.model.val_true_seg_msrs).flatten(),
            pred=np.array(self.model.val_pred_seg_msrs).flatten(),
            save_file=self.log_dir / f'validation/plots/hit_rate_plots/step_{epoch}.png',
            logger=self.logger
        )
        val_rnd_idx = np.random.randint(0, len(self.model.val_true_seg_msrs)-1)
        monitor_seg_error(
            ground_truth=self.model.val_gt_msks[train_rnd_idx],
            prediction=self.model.val_aug_msks[train_rnd_idx],
            seg_measures=self.model.val_gt_seg_msrs[train_rnd_idx],
            save_dir=f'val/plots/error_monitor/epoch_{epoch}'
        )

        # # Clean the data for the next epoch
        # self.model.train_true_seg_msrs = []
        # self.model.train_pred_seg_msrs = []
        # self.model.train_gt_msks = []
        # self.model.train_aug_msks = []
        #
        # self.model.val_true_seg_msrs = []
        # self.model.val_pred_seg_msrs = []
        # self.model.val_gt_msks = []
        # self.model.val_aug_msks = []

