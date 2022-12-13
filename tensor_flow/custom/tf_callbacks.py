import os
import pathlib
import tensorflow as tf
import numpy as np
import logging

from global_configs.general_configs import HR_AET_PERCENTAGE, HR_AET_FIGSIZE
from utils.aux_funcs import (
    scatter_plot,
    hit_rate_plot,
    to_numpy,
    err_log, save_figure, get_image_from_figure, plot_image_mask, float_2_str,
)


def write_figure_to_tensorboard(writer, figure, tag: str, step: int):
    with tf.device('/cpu:0'):
        with writer.as_default():
            # -> Write the scatter plot
            tf.summary.image(
                tag,
                get_image_from_figure(figure=figure),
                step=step
            )


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: pathlib.Path or str, tensorboard_logs: bool = False, logger: logging.Logger = None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.train_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'train'))
        self.val_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'validation'))

        self.tb_logs = tensorboard_logs

        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        try:
            # TRAIN
            # ----------------
            # - Scatter plot -
            # ----------------
            train_scatter_plots_dir = self.log_dir / f'train/plots/scatter'
            scatter_plot(
                x=np.array(self.model.train_epch_gt_seg_msrs).flatten(),
                y=np.array(self.model.train_epch_pred_seg_msrs).flatten(),
                save_file=train_scatter_plots_dir / f'step_{epoch}.png',
                tensorboard_params=dict(
                   writer=self.train_file_writer,
                   tag='1 - Scatter Plot',
                   step=epoch
                ) if self.tb_logs else None,
                logger=self.logger
            )

            # * Save metadata
            to_numpy(data=np.array(self.model.train_epch_gt_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.train_epch_pred_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # -----------------
            # - Hit rate plot -
            # -----------------
            train_hit_rate_plots_dir = self.log_dir / f'train/plots/hit_rate'
            train_hit_rate_hist, train_hit_rate_bins = hit_rate_plot(
                true=np.array(self.model.train_epch_gt_seg_msrs).flatten(),
                pred=np.array(self.model.train_epch_pred_seg_msrs).flatten(),
                hit_rate_percent=HR_AET_PERCENTAGE,
                figsize=HR_AET_FIGSIZE,
                save_file=train_hit_rate_plots_dir / f'step_{epoch}.png',
                tensorboard_params=dict(
                    writer=self.train_file_writer,
                    tag='2 - Hit Rate',
                    step=epoch
                ) if self.tb_logs else None,
                logger=self.logger
            )

            # * Save metadata
            to_numpy(data=train_hit_rate_hist, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=train_hit_rate_bins, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # -----------
            # - Samples -
            # -----------
            train_img = self.model.train_btch_smpl_dict.get('image')
            train_msk = self.model.train_btch_smpl_dict.get('mask')
            train_true_sm = self.model.train_btch_smpl_dict.get('true_seg_measure')
            train_pred_sm = self.model.train_btch_smpl_dict.get('pred_seg_measure')
            plot_image_mask(
                image=train_img,
                mask=train_msk,
                suptitle='Image with Corresponding Mask (red)',
                title=f'Seg measure: true - {train_true_sm:.4f}, pred - {train_pred_sm:.4f}',
                figsize=(20, 20),
                tensorboard_params=dict(
                    writer=self.train_file_writer,
                    tag='3 - Sample',
                    step=epoch
                ) if self.tb_logs else None,
                save_file=self.log_dir / f'train/samples/{epoch}_true_{float_2_str(train_true_sm)}_pred_{float_2_str(train_pred_sm)}.png',
            )

            # VALIDATION
            # ----------------
            # - Scatter plot -
            # ----------------
            val_scatter_plots_dir = self.log_dir / f'validation/plots/scatter'
            scatter_plot(
                x=np.array(self.model.val_epch_gt_seg_msrs).flatten(),
                y=np.array(self.model.val_epch_pred_seg_msrs).flatten(),
                save_file=val_scatter_plots_dir / f'step_{epoch}.png',
                tensorboard_params=dict(
                    writer=self.val_file_writer,
                    tag='1 - Scatter Plot',
                    step=epoch
                ) if self.tb_logs else None,
                logger=self.logger
            )

            # * Save metadata
            to_numpy(data=np.array(self.model.val_epch_gt_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.val_epch_pred_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # -----------------
            # - Hit rate plot -
            # -----------------
            val_hit_rate_plots_dir = self.log_dir / f'validation/plots/hit_rate'
            val_hit_rate_hist, val_hit_rate_bins = hit_rate_plot(
                true=np.array(self.model.val_epch_gt_seg_msrs).flatten(),
                pred=np.array(self.model.val_epch_pred_seg_msrs).flatten(),
                hit_rate_percent=HR_AET_PERCENTAGE,
                figsize=HR_AET_FIGSIZE,
                save_file=val_hit_rate_plots_dir / f'step_{epoch}.png',
                tensorboard_params=dict(
                    writer=self.val_file_writer,
                    tag='2 - Hit Rate',
                    step=epoch
                ) if self.tb_logs else None,
                logger=self.logger
            )

            # * Save metadata
            to_numpy(data=val_hit_rate_hist, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=val_hit_rate_bins, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # ----------
            # - Sample -
            # ----------
            val_img = self.model.val_btch_smpl_dict.get('image')
            val_msk = self.model.val_btch_smpl_dict.get('mask')
            val_true_sm = self.model.val_btch_smpl_dict.get('true_seg_measure')
            val_pred_sm = self.model.val_btch_smpl_dict.get('pred_seg_measure')
            plot_image_mask(
                image=val_img,
                mask=val_msk,
                suptitle='Image with Corresponding Mask (red)',
                title=f'Seg measure: true - {val_true_sm:.4f}, pred - {val_pred_sm:.4f}',
                figsize=(20, 20),
                tensorboard_params=dict(
                    writer=self.val_file_writer,
                    tag='3 - Sample',
                    step=epoch
                ) if self.tb_logs else None,
                save_file=self.log_dir / f'validation/samples/{epoch}_true_{float_2_str(val_true_sm)}_pred_{float_2_str(val_pred_sm)}.png',
            )

        except RuntimeError as err:
            err_log(logger=self.logger, message=f'{err}')

        # - Clean the data for the next epoch
        self.model.train_epch_gt_seg_msrs = np.array([])
        self.model.train_epch_pred_seg_msrs = np.array([])

        self.model.val_epch_gt_seg_msrs = np.array([])
        self.model.val_epch_pred_seg_msrs = np.array([])
