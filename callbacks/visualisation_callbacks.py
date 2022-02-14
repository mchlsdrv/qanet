import pathlib
import tensorflow as tf
import numpy as np
from utils.visualisation_utils.plotting_funcs import (
    plot_scatter
)

from utils.image_utils.image_aux import (
    get_image_from_figure
)


# - CLASSES
class ScatterPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, figsize: tuple = (20, 10), log_dir: pathlib.Path = None, log_interval: int = 10):
        super().__init__()
        self.log_dir = log_dir
        self.train_file_writer = tf.summary.create_file_writer(self.log_dir / 'train')
        self.val_file_writer = tf.summary.create_file_writer(self.log_dir / 'val')
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
        if epoch % self.log_interval == 0:
            print(f'\nSaving scatter plot of the seg measures for epoch #{epoch} to: \'{self.log_dir}\'...')

            # 5) Plot
            with self.train_file_writer.as_default():
                save_file = None
                if isinstance(self.logdir, pathlib.Path):
                    save_file = self.log_dir / f'train_epoch_{epoch}.png'

                fig = plot_scatter(
                    true_seg_measures=self.model.train_epoch_trgt_seg_msrs,
                    predicted_seg_measures=self.model.train_epoch_pred_seg_msrs,
                    figsize=self.figsize,
                    save_file=save_file
                )

                tf.summary.image(
                    f'Train (epoch #{epoch})',
                    get_image_from_figure(figure=fig),
                    step=epoch
                )

            with self.val_file_writer.as_default():
                save_file = None
                if isinstance(self.logdir, pathlib.Path):
                    save_file = self.log_dir / f'val_epoch_{epoch}.png'

                fig = plot_scatter(
                    true_seg_measures=self.model.val_epoch_trgt_seg_msrs,
                    predicted_seg_measures=self.model.val_epoch_pred_seg_msrs,
                    figsize=self.figsize,
                    save_file=save_file
                )

                tf.summary.image(
                    f'Validation (epoch #{epoch})',
                    get_image_from_figure(figure=fig),
                    step=epoch
                )

        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])
