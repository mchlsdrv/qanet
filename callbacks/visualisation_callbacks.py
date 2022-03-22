import pathlib
import tensorflow as tf
import numpy as np
import logging
# from utils.visualisation_utils.plotting_funcs import (
from utils.plotting_funcs import (
    plot_scatter,
    plot
)
from configs.general_configs import (
    DEBUG_LEVEL,
    PLOT_OUTLIERS,
    N_OUTLIERS
)
from utils.image_utils.image_aux import (
    get_image_from_figure,
)
from utils import aux_funcs


# - CLASSES
class TrainLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, figsize: tuple = (20, 10), log_dir: pathlib.Path = None, log_interval: int = 10, logger: logging.Logger = None):
        super().__init__()
        self.log_dir = log_dir
        self.train_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'train'))
        self.val_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'validation'))
        self.logger = logger
        self.figsize = figsize
        self.log_interval = log_interval

    def on_training_begin(self, logs=None):
        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])

    def write_images_to_tensorboard(self, writer, data: dict, step: int, save_file: pathlib.Path = None):
        with writer.as_default():
            with tf.device('/cpu:0'):
                # -> Write the scatter plot
                tf.summary.image(
                    '1 - Scatter',
                    get_image_from_figure(
                        figure=plot_scatter(
                            x=data.get('Scatter')['x'],
                            y=data.get('Scatter')['y'],
                            save_file=save_file
                        )
                    ),
                    step=step
                )

                # - Write the images
                # -> Normalize the images
                imgs = data.get('Images')
                disp_imgs = imgs - tf.reduce_min(imgs, axis=(1, 2, 3), keepdims=True)
                disp_imgs = disp_imgs / tf.reduce_max(disp_imgs, axis=(1, 2, 3), keepdims=True)
                tf.summary.image(
                    '2 - Images',
                    disp_imgs,
                    max_outputs=1,
                    step=step
                )

                # -> Write the segmentations
                disp_segs = data.get('Segmentations')
                tf.summary.image(
                    '3 - Segmentations',
                    disp_segs,
                    max_outputs=1,
                    step=step
                )

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            aux_funcs.info_log(logger=self.logger, message=f'\nSaving scatter plot of the seg measures for epoch #{epoch} to: \'{self.log_dir}\'...')

            # - Write to tensorboard
            # -- Train log
            self.write_images_to_tensorboard(
                writer=self.train_file_writer,
                data=dict(
                    Images=self.model.train_imgs,
                    Segmentations=self.model.train_aug_segs,
                    Scatter=dict(
                        x=self.model.train_epoch_trgt_seg_msrs,
                        y=self.model.train_epoch_pred_seg_msrs,
                        save_file=self.log_dir / f'train/plots/scatter_plot_step_{epoch}.png'
                    )
                ),
                step=epoch
            )

            # -- Validation log
            self.write_images_to_tensorboard(
                writer=self.val_file_writer,
                data=dict(
                    Images=self.model.val_imgs,
                    Segmentations=self.model.val_aug_segs,
                    Scatter=dict(
                        x=self.model.val_epoch_trgt_seg_msrs,
                        y=self.model.val_epoch_pred_seg_msrs,
                        save_file=self.log_dir / f'validation/plots/scatter_plot_step_{epoch}.png'
                    )
                ),
                step=epoch
            )

            # - Save the outlier images locally
            if PLOT_OUTLIERS:
                aux_funcs.info_log(logger=self.logger, message=f'Adding {N_OUTLIERS} outlier train and validation plots to {self.log_dir} directory...')
                for idx, outlier in enumerate(self.model.train_epoch_outliers):
                    plot(
                        images=[outlier[0], outlier[1]],
                        labels=['', ''],
                        suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}',
                        save_file=self.log_dir / f'train/outliers/epoch_{epoch}_{idx}.png'
                    )
                    if idx > N_OUTLIERS:
                        break

                for idx, outlier in enumerate(self.model.val_epoch_outliers):
                    plot(
                        images=[outlier[0], outlier[1]],
                        labels=['', ''],
                        suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}',
                        save_file=self.log_dir / f'validation/outliers/epoch_{epoch}_{idx}.png'
                    )
                    if idx > N_OUTLIERS:
                        break

        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.train_epoch_outliers= list()
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_outliers= list()
