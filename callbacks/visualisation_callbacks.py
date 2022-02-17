import pathlib
import tensorflow as tf
import numpy as np
from utils.visualisation_utils.plotting_funcs import (
    plot_scatter
)

from utils.image_utils.image_aux import (
    get_image_from_figure,
)
from configs.general_configs import (
    SCATTER_PLOT_FIGSIZE
)
# from utils.general_utils.aux_funcs import (
#     write_images_to_tensorboard
# )

def write_images_to_tensorboard(writer, data: dict, step: int):
    with writer.as_default():
        with tf.device('/cpu:0'):
            # - Write the images
            # -> Normalize the images
            imgs = data.get('Images')
            disp_imgs = imgs - tf.reduce_min(imgs, axis=(1, 2, 3), keepdims=True)
            disp_imgs = disp_imgs / tf.reduce_max(disp_imgs, axis=(1, 2, 3), keepdims=True)
            tf.summary.image(
                'Images',
                disp_imgs,
                max_outputs=1,
                step=step
            )

            # -> Write the segmentations
            disp_segs = data.get('Segmentations')
            tf.summary.image(
                'Segmentations',
                disp_segs,
                max_outputs=1,
                step=step
            )

            # -> Write the scatter plot
            tf.summary.image(
                'Scatter',
                get_image_from_figure(
                    figure=plot_scatter(
                        x=data.get('Scatter')['x'],
                        y=data.get('Scatter')['y'],
                        figsize=data.get('Scatter')['figsize'],
                        save_file=None
                    )
                ),
                step=step
            )

# - CLASSES
class ScatterPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, figsize: tuple = (20, 10), log_dir: pathlib.Path = None, log_interval: int = 10):
        super().__init__()
        self.log_dir = log_dir
        self.train_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'train'))
        self.val_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'val'))
        self.figsize = figsize
        self.log_interval = log_interval

    def on_training_begin(self, logs=None):
        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])

    def on_epoch_end(self, epoch, logs=None):
        # 1) Get the layers
        # print(self.model.train_epoch_trgt_seg_msrs)
        # print(self.model.train_epoch_pred_seg_msrs)
        # print(self.model.val_epoch_trgt_seg_msrs)
        # print(self.model.val_epoch_pred_seg_msrs)
        if epoch % self.log_interval == 0:
            print(f'\nSaving scatter plot of the seg measures for epoch #{epoch} to: \'{self.log_dir}\'...')

            # 5) Plot
            write_images_to_tensorboard(
                writer=self.train_file_writer,
                data=dict(
                    Images=self.model.train_imgs,
                    Segmentations=self.model.train_pred_segs,
                    Scatter=dict(
                        x=self.model.train_epoch_trgt_seg_msrs,
                        y=self.model.train_epoch_pred_seg_msrs,
                        figsize=self.figsize
                    )
                ),
                step=epoch
            )
            write_images_to_tensorboard(
                writer=self.val_file_writer,
                data=dict(
                    Images=self.model.val_imgs,
                    Segmentations=self.model.val_pred_segs,
                    Scatter=dict(
                        x=self.model.val_epoch_trgt_seg_msrs,
                        y=self.model.val_epoch_pred_seg_msrs,
                        figsize=self.figsize
                    )
                ),
                step=epoch
            )
            # with self.train_file_writer.as_default():
            #     save_file = None
            #     if isinstance(self.log_dir, pathlib.Path):
            #         save_file = self.log_dir / f'train_epoch_{epoch}.png'
            #
            #     fig = plot_scatter(
            #         x=self.model.train_epoch_trgt_seg_msrs,
            #         y=self.model.train_epoch_pred_seg_msrs,
            #         figsize=self.figsize,
            #         save_file=save_file
            #     )
            #
            #     tf.summary.image(
            #         f'Train (epoch #{epoch})',
            #         get_image_from_figure(figure=fig),
            #         step=epoch
            #     )
            #
            # with self.val_file_writer.as_default():
            #     save_file = None
            #     if isinstance(self.log_dir, pathlib.Path):
            #         save_file = self.log_dir / f'val_epoch_{epoch}.png'
            #
            #     fig = plot_scatter(
            #         x=self.model.val_epoch_trgt_seg_msrs,
            #         y=self.model.val_epoch_pred_seg_msrs,
            #         figsize=self.figsize,
            #         save_file=save_file
            #     )
            #
            #     tf.summary.image(
            #         f'Validation (epoch #{epoch})',
            #         get_image_from_figure(figure=fig),
            #         step=epoch
            #     )

        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])
