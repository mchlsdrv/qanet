import os
import io
import pathlib

import tensorflow as tf
import numpy as np
import logging

import wandb
from matplotlib import pyplot as plt

from global_configs.general_configs import HR_AET_PERCENTAGE, HR_AET_FIGSIZE
from utils.aux_funcs import (
    to_numpy,
    err_log,
)
from utils.visual_funcs import (
    save_figure,
    get_rgb_mask_figure,
    get_hit_rate_plot_figure,
    get_scatter_plot_figure,
    get_image_figure,
)


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    # plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_figure_to_tensorboard(writer, figure, tag: str, step: int):
    with tf.device('/cpu:0'):
        with writer.as_default():
            # -> Write the scatter plot
            tf.summary.image(
                tag,
                get_image_from_figure(figure=figure),
                step=step
            )


def write_scalar_to_tensorboard(writer, value, tag: str, step: int):
    with tf.device('/cpu:0'):
        with writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: pathlib.Path or str, tensorboard_logs: bool = False, wandb_logs: bool = False,
                 logger: logging.Logger = None):
        super().__init__()

        # - Create the log dir
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # - Create the train file writer
        self.train_file_writer = tf.summary.create_file_writer(
            str(self.log_dir / 'train'))

        # - Create the train file writer
        self.val_file_writer = tf.summary.create_file_writer(
            str(self.log_dir / 'validation'))

        # - Create the train scatter plots directory
        self.train_scatter_plots_dir = self.log_dir / f'train/plots/scatter'
        os.makedirs(self.train_scatter_plots_dir, exist_ok=True)

        # - Create the train hit rate plots directory
        self.train_hit_rate_plots_dir = self.log_dir / f'train/plots/hit_rate'
        os.makedirs(self.train_hit_rate_plots_dir, exist_ok=True)

        # - Create the train samples directory
        self.train_sample_dir = self.log_dir / f'train/plots/samples'
        os.makedirs(self.train_sample_dir, exist_ok=True)

        # - Create the val scatter plots directory
        self.val_scatter_plots_dir = self.log_dir / f'validation/plots/scatter'
        os.makedirs(self.val_scatter_plots_dir, exist_ok=True)

        # - Create the train hit rate plots directory
        self.val_hit_rate_plots_dir = \
            self.log_dir / f'validation/plots/hit_rate'
        os.makedirs(self.val_hit_rate_plots_dir, exist_ok=True)

        # - Create the train samples directory
        self.val_sample_dir = self.log_dir / f'validation/plots/samples'
        os.makedirs(self.val_sample_dir, exist_ok=True)

        # - History arrays
        self.train_losses = np.array([])
        self.val_losses = np.array([])

        self.train_rhos = np.array([])
        self.val_rhos = np.array([])

        self.train_mses = np.array([])
        self.val_mses = np.array([])

        self.lrs = np.array([])

        self.tb_logs = tensorboard_logs

        self.wandb_logs = wandb_logs

        self.logger = logger

    def log_figure(self, figure, file_writer, step: int, tag: str, procedure: str, save_file: pathlib.Path or str):
        save_figure(
            figure=figure,
            save_file=save_file,
            close_figure=False,
            logger=self.logger
        )

        if self.tb_logs:
            write_figure_to_tensorboard(
                writer=file_writer,
                figure=figure,
                tag=tag,
                step=step
            )

        if self.wandb_logs:
            wandb.log(
                {
                    f'{tag}-{procedure}': wandb.Image(
                        get_image_from_figure(figure=figure))
                }
            )

        plt.close(figure)

    def on_epoch_end(self, epoch, logs=None):
        try:
            # TRAIN
            # ----------------
            # - Scatter plot -
            # ----------------
            train_sctr_fig, train_rho, train_p, train_mse = \
                get_scatter_plot_figure(
                    x=np.array(self.model.train_epch_gt_seg_msrs).flatten(),
                    y=np.array(self.model.train_epch_pred_seg_msrs).flatten(),
                    plot_type='train',
                    logger=self.logger
                )

            self.train_rhos, self.train_mses = \
                np.append(self.train_rhos, train_rho), np.append(self.train_mses, train_mse)

            self.log_figure(
                figure=train_sctr_fig,
                file_writer=self.train_file_writer,
                step=epoch,
                tag='1 - Scatter Plot',
                procedure='train',
                save_file=self.train_scatter_plots_dir / f'step_{epoch}.png'
            )

            # * Save metadata
            to_numpy(
                data=np.array(self.model.train_epch_gt_seg_msrs).flatten(),
                file_path=self.train_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy',
                overwrite=False, logger=self.logger)
            to_numpy(
                data=np.array(self.model.train_epch_pred_seg_msrs).flatten(),
                file_path=self.train_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy',
                overwrite=False, logger=self.logger)

            # -------------------------
            # - Scalars - Rho and MSE -
            # -------------------------
            write_scalar_to_tensorboard(
                writer=self.train_file_writer,
                value=train_rho,
                tag='Rho',
                step=epoch
            )

            write_scalar_to_tensorboard(
                writer=self.train_file_writer,
                value=train_mse,
                tag='MSE',
                step=epoch
            )

            # -----------
            # - Samples -
            # -----------
            train_img = self.model.train_btch_smpl_dict.get(
                'image').astype(np.float32)
            train_msk = self.model.train_btch_smpl_dict.get(
                'mask').astype(np.float32)
            train_true_sm = self.model.train_btch_smpl_dict.get(
                'true_seg_measure').astype(np.float32)
            train_pred_sm = self.model.train_btch_smpl_dict.get(
                'pred_seg_measure').astype(np.float32)
            # - Mask
            train_msk_fig = get_rgb_mask_figure(
                mask=train_msk,
                suptitle='Mask',
                title=f'Seg measure: true - {train_true_sm:.4f}, '
                      f'pred - {train_pred_sm:.4f}',
                figsize=(20, 20),
            )
            self.log_figure(
                figure=train_msk_fig,
                file_writer=self.train_file_writer,
                step=epoch,
                tag='2 - Mask',
                procedure='train',
                save_file=self.train_sample_dir / f'step_{epoch}.png'
            )

            # - Image
            train_img_fig = get_image_figure(
                image=train_img,
                suptitle='Image',
                title='',
                figsize=(20, 20),
            )
            self.log_figure(
                figure=train_img_fig,
                file_writer=self.train_file_writer,
                step=epoch,
                tag='3 - Image',
                procedure='train',
                save_file=self.train_sample_dir / f'step_{epoch}.png'
            )

            # -----------
            # - Outlier -
            # -----------
            train_img = self.model.train_btch_outlier_smpl_dict.get('image')
            train_msk = self.model.train_btch_outlier_smpl_dict.get('mask')
            train_true_sm = self.model.train_btch_outlier_smpl_dict.get('true_seg_measure')
            train_pred_sm = self.model.train_btch_outlier_smpl_dict.get('pred_seg_measure')

            if train_img is not None and train_msk is not None and \
                    train_true_sm is not None and train_pred_sm is not None:
                train_img = train_img.astype(np.float32)
                train_msk = train_msk.astype(np.float32)
                train_true_sm = train_true_sm.astype(np.float32)
                train_pred_sm = train_pred_sm.astype(np.float32)
                # - Mask
                train_msk_fig = get_rgb_mask_figure(
                    mask=train_msk,
                    suptitle='Outlier Mask',
                    title=f'Seg measure: true - {train_true_sm:.4f}, '
                          f'pred - {train_pred_sm:.4f}',
                    figsize=(20, 20),
                )
                self.log_figure(
                    figure=train_msk_fig,
                    file_writer=self.train_file_writer,
                    step=epoch,
                    tag='4 - Outlier Mask',
                    procedure='train',
                    save_file=self.train_sample_dir / f'step_{epoch}.png'
                )

                # - Image
                train_img_fig = get_image_figure(
                    image=train_img,
                    suptitle='Outlier Image',
                    title='',
                    figsize=(20, 20),
                )
                self.log_figure(
                    figure=train_img_fig,
                    file_writer=self.train_file_writer,
                    step=epoch,
                    tag='5 - Outlier Image',
                    procedure='train',
                    save_file=self.train_sample_dir / f'step_{epoch}.png'
                )

            # -----------------
            # - Hit rate plot -
            # -----------------
            # train_hit_rate_fig, train_hit_rate_hist, train_hit_rate_bins = \
            #     get_hit_rate_plot_figure(
            #         true=np.array(self.model.train_epch_gt_seg_msrs).flatten(),
            #         pred=np.array(
            #             self.model.train_epch_pred_seg_msrs).flatten(),
            #         hit_rate_percent=HR_AET_PERCENTAGE,
            #         figsize=HR_AET_FIGSIZE,
            #         logger=self.logger
            #     )
            #
            # self.log_figure(
            #     figure=train_hit_rate_fig,
            #     file_writer=self.train_file_writer,
            #     step=epoch,
            #     tag='4 - Hit Rate',
            #     procedure='train',
            #     save_file=self.train_hit_rate_plots_dir / f'step_{epoch}.png'
            # )
            #
            # # * Save metadata
            # to_numpy(
            #     data=train_hit_rate_hist,
            #     file_path=self.train_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy',
            #     overwrite=False, logger=self.logger)
            # to_numpy(
            #     data=train_hit_rate_bins,
            #     file_path=self.train_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy',
            #     overwrite=False, logger=self.logger)

            # -----------------
            # - Learning Rate -
            # -----------------
            write_scalar_to_tensorboard(
                writer=self.train_file_writer,
                value=self.model.optimizer.learning_rate.numpy(),
                tag='Learning Rate',
                step=epoch
            )

            # VALIDATION
            # ----------------
            # - Scatter plot -
            # ----------------
            val_sctr_fig, val_rho, val_p, val_mse = get_scatter_plot_figure(
                x=np.array(self.model.val_epch_gt_seg_msrs).flatten(),
                y=np.array(self.model.val_epch_pred_seg_msrs).flatten(),
                plot_type='val',
                logger=self.logger
            )

            self.val_rhos, self.val_mses = np.append(self.val_rhos, val_rho), np.append(self.val_mses, val_mse)

            self.log_figure(
                figure=val_sctr_fig,
                file_writer=self.val_file_writer,
                step=epoch,
                tag='1 - Scatter Plot',
                procedure='val',
                save_file=self.val_scatter_plots_dir / f'step_{epoch}.png'
            )

            # * Save metadata
            to_numpy(
                data=np.array(self.model.val_epch_gt_seg_msrs).flatten(),
                file_path=self.val_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy',
                overwrite=False, logger=self.logger)
            to_numpy(
                data=np.array(self.model.val_epch_pred_seg_msrs).flatten(),
                file_path=self.val_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy',
                overwrite=False, logger=self.logger)

            # -------------------------
            # - Scalars - Rho and MSE -
            # -------------------------
            write_scalar_to_tensorboard(
                writer=self.val_file_writer,
                value=val_rho,
                tag='Rho',
                step=epoch
            )

            write_scalar_to_tensorboard(
                writer=self.val_file_writer,
                value=val_mse,
                tag='MSE',
                step=epoch
            )

            # ----------
            # - Sample -
            # ----------
            val_img = self.model.val_btch_smpl_dict.get(
                'image').astype(np.float32)
            val_msk = self.model.val_btch_smpl_dict.get(
                'mask').astype(np.float32)
            val_true_sm = self.model.val_btch_smpl_dict.get(
                'true_seg_measure').astype(np.float32)
            val_pred_sm = self.model.val_btch_smpl_dict.get(
                'pred_seg_measure').astype(np.float32)

            # - Mask
            val_msk_fig = get_rgb_mask_figure(
                mask=val_msk,
                suptitle='Mask',
                title=f'Seg measure: true - {val_true_sm:.4f}, '
                      f'pred - {val_pred_sm:.4f}',
                figsize=(20, 20),
            )
            self.log_figure(
                figure=val_msk_fig,
                file_writer=self.val_file_writer,
                step=epoch,
                tag='2 - Mask',
                procedure='val',
                save_file=self.val_sample_dir / f'step_{epoch}.png'
            )

            # - Image
            val_img_fig = get_image_figure(
                image=val_img,
                suptitle='Image',
                title='',
                figsize=(20, 20),
            )
            self.log_figure(
                figure=val_img_fig,
                file_writer=self.val_file_writer,
                step=epoch,
                tag='3 - Image',
                procedure='val',
                save_file=self.val_sample_dir / f'step_{epoch}.png'
            )

            # -----------
            # - Outlier -
            # -----------
            val_img = self.model.val_btch_outlier_smpl_dict.get('image')
            val_msk = self.model.val_btch_outlier_smpl_dict.get('mask')
            val_true_sm = self.model.val_btch_outlier_smpl_dict.get('true_seg_measure')
            val_pred_sm = self.model.val_btch_outlier_smpl_dict.get('pred_seg_measure')
            if val_img is not None and val_msk is not None and \
                    val_true_sm is not None and val_pred_sm is not None:
                val_img = val_img.astype(np.float32)
                val_msk = val_msk.astype(np.float32)
                val_true_sm = val_true_sm.astype(np.float32)
                val_pred_sm = val_pred_sm.astype(np.float32)

                # - Mask
                val_msk_fig = get_rgb_mask_figure(
                    mask=val_msk,
                    suptitle='Outlier Mask',
                    title=f'Seg measure: true - {val_true_sm:.4f}, '
                          f'pred - {val_pred_sm:.4f}',
                    figsize=(20, 20),
                )
                self.log_figure(
                    figure=val_msk_fig,
                    file_writer=self.val_file_writer,
                    step=epoch,
                    tag='4 - Outlier Mask',
                    procedure='val',
                    save_file=self.val_sample_dir / f'step_{epoch}.png'
                )

                # - Image
                val_img_fig = get_image_figure(
                    image=val_img,
                    suptitle='Outlier Image',
                    title='',
                    figsize=(20, 20),
                )
                self.log_figure(
                    figure=val_img_fig,
                    file_writer=self.val_file_writer,
                    step=epoch,
                    tag='5 - Outlier Image',
                    procedure='val',
                    save_file=self.val_sample_dir / f'step_{epoch}.png'
                )

            # -----------------
            # - Hit rate plot -
            # -----------------
            # val_hit_rate_fig, val_hit_rate_hist, val_hit_rate_bins = \
            #     get_hit_rate_plot_figure(
            #         true=np.array(self.model.val_epch_gt_seg_msrs).flatten(),
            #         pred=np.array(self.model.val_epch_pred_seg_msrs).flatten(),
            #         hit_rate_percent=HR_AET_PERCENTAGE,
            #         figsize=HR_AET_FIGSIZE,
            #         logger=self.logger
            #     )
            # self.log_figure(
            #     figure=val_hit_rate_fig,
            #     file_writer=self.val_file_writer,
            #     step=epoch,
            #     tag='4 - Hit Rate',
            #     procedure='val',
            #     save_file=self.val_hit_rate_plots_dir / f'step_{epoch}.png'
            # )
            #
            # # * Save metadata
            # to_numpy(
            #     data=val_hit_rate_hist,
            #     file_path=self.val_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy',
            #     overwrite=False, logger=self.logger)
            # to_numpy(
            #     data=val_hit_rate_bins,
            #     file_path=self.val_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy',
            #     overwrite=False, logger=self.logger)

            # ----------------------
            # - Weights and Biases -
            # ----------------------
            if self.wandb_logs:
                wandb.log({
                    'train loss': self.model.train_epch_losses.mean(),
                    'val loss': self.model.val_epch_losses.mean(),
                    'train rho': train_rho,
                    'val rho': val_rho,
                    'train mse': train_mse,
                    'val mse': val_mse,
                    'learning rate': self.model.optimizer.learning_rate.numpy()
                }
                )

        except RuntimeError as err:
            err_log(logger=self.logger, message=f'{err}')

        # - Clean the data for the next epoch
        self.model.train_epch_gt_seg_msrs = np.array([])
        self.model.train_epch_pred_seg_msrs = np.array([])

        self.model.val_epch_gt_seg_msrs = np.array([])
        self.model.val_epch_pred_seg_msrs = np.array([])
