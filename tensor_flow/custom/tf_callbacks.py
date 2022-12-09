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
    err_log, save_figure, get_image_from_figure,
)

# from utils.aux_funcs import (
#     get_image_from_figure,
# )

# this is the order in which my classes will be displayed
# this is a reverse map of the integer class id to the string class label


# - CLASSES
# class ProgressLogCallback(tf.keras.callbacks.Callback):
#     def __init__(self, log_type: str, figsize: tuple = (20, 10), log_dir: pathlib.Path = None, log_interval: int = 10, logger: logging.Logger = None):
#         super().__init__()
#         self.log_dir = log_dir
#         self.train_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'train'))
#         self.val_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'validation'))
#         self.logger = logger
#         self.figsize = figsize
#         self.log_interval = log_interval
#         self.epoch = 0
#         self.end = False
#         self.log_type = log_type
#
#     @staticmethod
#     def write_images_to_tensorboard(writer, data: dict, step: int, save_file: pathlib.Path = None):
#         if data.get('Scatter')['x'].any() and data.get('Scatter')['y'].any():
#             with writer.as_default():
#                 with tf.device('/cpu:0'):
#                     # -> Write the scatter plot
#                     tf.summary.image(
#                         '1 - Scatter',
#                         get_image_from_figure(
#                             figure=plot_scatter(
#                                 x=data.get('Scatter')['x'],
#                                 y=data.get('Scatter')['y'],
#                                 save_file=save_file
#                             )
#                         ),
#                         step=step
#                     )
#
#                     # - Write the images
#                     # -> Normalize the images
#                     imgs = data.get('Images')
#                     if len(imgs.shape) < 4:
#                         imgs = np.expand_dims(imgs, -1)
#                     imgs = imgs - tf.reduce_min(imgs, axis=(1, 2, 3), keepdims=True)
#                     imgs = imgs / tf.reduce_max(imgs, axis=(1, 2, 3), keepdims=True)
#                     tf.summary.image(
#                         '2 - Images',
#                         imgs,
#                         max_outputs=1,
#                         step=step
#                     )
#
#                     # -> Write the segmentations
#                     segs = data.get('Segmentations')
#                     if len(segs.shape) < 4:
#                         segs = np.expand_dims(segs, -1)
#                     tf.summary.image(
#                         '3 - Segmentations',
#                         segs,
#                         max_outputs=1,
#                         step=step
#                     )
#
#     def on_training_begin(self, logs=None):
#         # - Clean the seg measures history arrays
#         self.model.train_epoch_trgt_seg_msrs = np.array([])
#         self.model.train_epoch_pred_seg_msrs = np.array([])
#         self.model.val_epoch_trgt_seg_msrs = np.array([])
#         self.model.val_epoch_pred_seg_msrs = np.array([])
#
#     # def on_training_end(self, logs=None):
#     #     self.end = True
#     #     self.on_epoch_end(epoch=self.epoch)
#
#     def on_test_begin(self, logs=None):
#         # - Clean the seg measures history arrays
#         self.model.val_epoch_trgt_seg_msrs = np.array([])
#         self.model.val_epoch_pred_seg_msrs = np.array([])
#
#     # def on_test_end(self, logs=None):
#     #     self.on_epoch_end(epoch=self.epoch)
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch = epoch
#         if epoch % self.log_interval == 0 or self.end or self.log_type == 'test':
#             aux_funcs.info_log(logger=self.logger, message=f'\nSaving scatter plot of the seg measures for epoch #{epoch} to: \'{self.log_dir}\'...')
#
#             # - Write to tensorboard
#             # -- Train log
#             if self.log_type == 'train':
#                 self.write_images_to_tensorboard(
#                     writer=self.train_file_writer,
#                     data=dict(
#                         Images=self.model.train_imgs,
#                         Segmentations=self.model.train_aug_segs,
#                         Scatter=dict(
#                             x=self.model.train_epoch_trgt_seg_msrs,
#                             y=self.model.train_epoch_pred_seg_msrs,
#                             save_file=self.log_dir / f'train/plots/scatter_plot_step_{epoch}.png'
#                         )
#                     ),
#                     step=epoch
#                 )
#
#             # -- Validation log
#             self.write_images_to_tensorboard(
#                 writer=self.val_file_writer,
#                 data=dict(
#                     Images=self.model.val_imgs,
#                     Segmentations=self.model.val_aug_segs,
#                     Scatter=dict(
#                         x=self.model.val_epoch_trgt_seg_msrs,
#                         y=self.model.val_epoch_pred_seg_msrs,
#                         save_file=self.log_dir / f'validation/plots/scatter_plot_step_{epoch}.png' if self.log_type == 'train' else self.log_dir / f'test/plots/scatter_plot_step_{epoch}.png'
#                     )
#                 ),
#                 step=epoch
#             )
#
#             # - Save the outlier images locally
#             if PLOT_OUTLIERS:
#                 aux_funcs.info_log(logger=self.logger, message=f'Adding {N_OUTLIERS} outlier train and validation plots to {self.log_dir} directory...')
#                 if self.log_type == 'train':
#                     for idx, outlier in enumerate(self.model.train_epoch_outliers):
#                         plot(
#                             images=[outlier[0], outlier[1]],
#                             labels=['', ''],
#                             suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}, Pixel Sum - {outlier[1].sum():.0f}',
#                             save_file=self.log_dir / f'train/outliers/epoch_{epoch}_{idx}.png'
#                         )
#                         if idx > N_OUTLIERS:
#                             break
#
#                 for idx, outlier in enumerate(self.model.val_epoch_outliers):
#                     plot(
#                         images=[outlier[0], outlier[1]],
#                         labels=['', ''],
#                         suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}, Pixel Sum - {outlier[1].sum()}',
#                         save_file=self.log_dir / f'validation/outliers/epoch_{epoch}_{idx}.png' if self.log_type == 'train' else self.log_dir / f'test/outliers/epoch_{epoch}_{idx}.png'
#                     )
#                     if idx > N_OUTLIERS:
#                         break
#
#             if PLOT_TRAIN_DATA_BATCHES:
#                 if self.model.train_loss_delta > LOSS_DELTA_TH:
#                     aux_funcs.info_log(logger=self.logger, message=f'Adding train data batches plots to {self.log_dir} directory...')
#                     if self.log_type == 'train':
#                         for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.model.train_imgs, self.model.train_aug_segs, self.model.train_trgt_seg_msrs, self.model.train_epch_pred_seg_msrs)):
#                             plot(
#                                 images=[img, seg],
#                                 labels=['', ''],
#                                 suptitle=f'Epoch: {epoch}, Seg Measures: Target - {trgt_seg_msr:.2f}, Predicted - {pred_seg_msr:.2f}, Pixel Sum - {seg.sum():.0f}',
#                                 save_file=self.log_dir / f'train/batch/epoch_{epoch}_{idx}.png'
#                             )
#
#             if PLOT_VALIDATION_DATA_BATCHES:
#                 if self.model.val_loss_delta > LOSS_DELTA_TH:
#                     aux_funcs.info_log(logger=self.logger, message=f'Adding validation data batches plots to {self.log_dir} directory...')
#                     for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.model.val_imgs, self.model.val_aug_segs, self.model.val_trgt_seg_msrs, self.model.val_epch_pred_seg_msrs)):
#                         plot(
#                             images=[img, seg],
#                             labels=['', ''],
#                             suptitle=f'Epoch: {epoch}, Seg Measures: Target - {trgt_seg_msr:.2f}, Predicted - {pred_seg_msr:.2f}, Pixel Sum - {seg.sum():.0f}',
#                             save_file=self.log_dir / f'validation/batch/epoch_{epoch}_{idx}.png'
#                         )
#
#         # - Clean the seg measures history arrays
#         self.model.train_epoch_trgt_seg_msrs = np.array([])
#         self.model.train_epoch_pred_seg_msrs = np.array([])
#         self.model.train_epoch_outliers = list()
#
#         self.model.val_epoch_trgt_seg_msrs = np.array([])
#         self.model.val_epoch_pred_seg_msrs = np.array([])
#         self.model.val_epoch_outliers = list()

# def write_images_to_tensorboard(writer, images: dict, step: int):
#     with writer.as_default():
#         with tf.device('/cpu:0'):
#             # -> Write the scatter plot
#             for img_name in images:
#                 fig = images.get(img_name)
#                 tf.summary.image(
#                     img_name,
#                     get_image_from_figure(figure=fig),
#                     step=step
#                 )
#                 # plt.close(fig)


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
            # - Scatter plot
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

            # - Save metadata
            to_numpy(data=np.array(self.model.train_epch_gt_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.train_epch_pred_seg_msrs).flatten(), file_path=train_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Hit rate plot
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

            # - Save metadata
            to_numpy(data=train_hit_rate_hist, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=train_hit_rate_bins, file_path=train_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # VALIDATION
            # - Scatter plot
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

            # - Save metadata
            to_numpy(data=np.array(self.model.val_epch_gt_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/gt_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=np.array(self.model.val_epch_pred_seg_msrs).flatten(), file_path=val_scatter_plots_dir / f'metadata/pred_seg_measures_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            # - Hit rate plot
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

            # - Save metadata
            to_numpy(data=val_hit_rate_hist, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_hist_epoch_{epoch}.npy', overwrite=False, logger=self.logger)
            to_numpy(data=val_hit_rate_bins, file_path=val_hit_rate_plots_dir / f'metadata/hit_rate_bins_epoch_{epoch}.npy', overwrite=False, logger=self.logger)

            if self.tb_logs:
                write_figure_to_tensorboard(
                    writer=self.train_file_writer,
                    figure=self.model.train_btch_smpl_fig,
                    tag='3 - Sample',
                    step=epoch
                )

                write_figure_to_tensorboard(
                    writer=self.val_file_writer,
                    figure=self.model.val_btch_smpl_fig,
                    tag='3 - Sample',
                    step=epoch
                )

            # - Save sample images
            save_figure(figure=self.model.train_btch_smpl_fig, save_file=self.log_dir / f'train/samples/{epoch}.png', logger=self.logger)
            save_figure(figure=self.model.val_btch_smpl_fig, save_file=self.log_dir / f'validation/samples/{epoch}.png', logger=self.logger)

        except RuntimeError as err:
            err_log(logger=self.logger, message=f'{err}')

        # - Clean the data for the next epoch
        self.model.train_epch_gt_seg_msrs = np.array([])
        self.model.train_epch_pred_seg_msrs = np.array([])

        self.model.val_epch_gt_seg_msrs = np.array([])
        self.model.val_epch_pred_seg_msrs = np.array([])
