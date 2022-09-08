import pathlib
import tensorflow as tf
import numpy as np
import logging
import wandb
from global_configs.general_configs import (
    PLOT_TRAIN_DATA_BATCHES,
    PLOT_VALIDATION_DATA_BATCHES,
    LOSS_DELTA_TH,
    PLOT_OUTLIERS,
    N_OUTLIERS
)
from utils.aux_funcs import (
    info_log,
    scatter_plot,
    show_images
)
from ..utils.tf_data_utils import (
    get_image_from_figure,
)


def log_masks(image, mask, true_seg_measure, pred_seg_measure):
    class_ids = np.unique(mask)[1:]
    class_labels = {}
    for cls_id in class_ids:
        if int(cls_id) > 0:
            class_labels[int(cls_id)] = f'Cell #{int(cls_id)}'
    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    mask_image = wandb.Image(
        image,
        masks={
            "predictions":
                {
                    "mask_data": mask,
                    "class_labels": class_labels
                }
            },
        # caption=f'Seg Measure ({procedure}): True - {true_seg_measure:.2f}, Predicted - {pred_seg_measure:.2f}'
        )
    return mask_image


# - CLASSES
class ProgressLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_type: str, figsize: tuple = (20, 10), log_dir: pathlib.Path = None, log_interval: int = 10, logger: logging.Logger = None):
        super().__init__()
        self.log_dir = log_dir
        self.train_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'train'))
        self.val_file_writer = tf.summary.create_file_writer(str(self.log_dir / 'validation'))
        self.logger = logger
        self.figsize = figsize
        self.log_interval = log_interval
        self.epoch = 0
        self.end = False
        self.log_type = log_type

    @staticmethod
    def write_images_to_tensorboard(writer, data: dict, step: int, save_file: pathlib.Path = None):
        if data.get('Scatter')['x'].any() and data.get('Scatter')['y'].any():
            with writer.as_default():
                with tf.device('/cpu:0'):
                    # -> Write the scatter plot
                    tf.summary.image(
                        '1 - Scatter',
                        get_image_from_figure(
                            figure=scatter_plot(
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
                    if len(imgs.shape) < 4:
                        imgs = np.expand_dims(imgs, -1)
                    imgs = imgs - tf.reduce_min(imgs, axis=(1, 2, 3), keepdims=True)
                    imgs = imgs / tf.reduce_max(imgs, axis=(1, 2, 3), keepdims=True)
                    tf.summary.image(
                        '2 - Images',
                        imgs,
                        max_outputs=1,
                        step=step
                    )

                    # -> Write the segmentations
                    segs = data.get('Segmentations')
                    if len(segs.shape) < 4:
                        segs = np.expand_dims(segs, -1)
                    tf.summary.image(
                        '3 - Segmentations',
                        segs,
                        max_outputs=1,
                        step=step
                    )

    def on_training_begin(self, logs=None):
        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])

    # def on_training_end(self, logs=None):
    #     self.end = True
    #     self.on_epoch_end(epoch=self.epoch)

    def on_test_begin(self, logs=None):
        # - Clean the seg measures history arrays
        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])

    # def on_test_end(self, logs=None):
    #     self.on_epoch_end(epoch=self.epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if epoch % self.log_interval == 0 or self.end or self.log_type == 'test':
            info_log(logger=self.logger, message=f'\nSaving scatter plot of the seg measures for epoch #{epoch} to: \'{self.log_dir}\'...')

            # - Write to tensorboard
            # -- Train log
            if self.log_type == 'train':
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
                        save_file=self.log_dir / f'validation/plots/scatter_plot_step_{epoch}.png' if self.log_type == 'train' else self.log_dir / f'test/plots/scatter_plot_step_{epoch}.png'
                    )
                ),
                step=epoch
            )

            # - Save the outlier images locally
            if PLOT_OUTLIERS:
                info_log(logger=self.logger, message=f'Adding {N_OUTLIERS} outlier train and validation plots to {self.log_dir} directory...')
                if self.log_type == 'train':
                    for idx, outlier in enumerate(self.model.train_epoch_outliers):
                        show_images(
                            images=[outlier[0], outlier[1]],
                            labels=['', ''],
                            suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}, Pixel Sum - {outlier[1].sum():.0f}',
                            save_file=self.log_dir / f'train/outliers/epoch_{epoch}_{idx}.png'
                        )
                        if idx > N_OUTLIERS:
                            break

                for idx, outlier in enumerate(self.model.val_epoch_outliers):
                    show_images(
                        images=[outlier[0], outlier[1]],
                        labels=['', ''],
                        suptitle=f'Epoch: {epoch}, Seg Measures: Target - {outlier[2]:.2f}, Predicted - {outlier[3]:.2f}, Pixel Sum - {outlier[1].sum()}',
                        save_file=self.log_dir / f'validation/outliers/epoch_{epoch}_{idx}.png' if self.log_type == 'train' else self.log_dir / f'test/outliers/epoch_{epoch}_{idx}.png'
                    )
                    if idx > N_OUTLIERS:
                        break

            if PLOT_TRAIN_DATA_BATCHES:
                if self.model.train_loss_delta > LOSS_DELTA_TH:
                    info_log(logger=self.logger, message=f'Adding train data batches plots to {self.log_dir} directory...')
                    if self.log_type == 'train':
                        for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.model.train_imgs, self.model.train_aug_segs, self.model.train_trgt_seg_msrs, self.model.train_pred_seg_msrs)):
                            show_images(
                                images=[img, seg],
                                labels=['', ''],
                                suptitle=f'Epoch: {epoch}, Seg Measures: Target - {trgt_seg_msr:.2f}, Predicted - {pred_seg_msr:.2f}, Pixel Sum - {seg.sum():.0f}',
                                save_file=self.log_dir / f'train/batch/epoch_{epoch}_{idx}.png'
                            )

            if PLOT_VALIDATION_DATA_BATCHES:
                if self.model.val_loss_delta > LOSS_DELTA_TH:
                    info_log(logger=self.logger, message=f'Adding validation data batches plots to {self.log_dir} directory...')
                    for idx, (img, seg, trgt_seg_msr, pred_seg_msr) in enumerate(zip(self.model.val_imgs, self.model.val_aug_segs, self.model.val_trgt_seg_msrs, self.model.val_pred_seg_msrs)):
                        show_images(
                            images=[img, seg],
                            labels=['', ''],
                            suptitle=f'Epoch: {epoch}, Seg Measures: Target - {trgt_seg_msr:.2f}, Predicted - {pred_seg_msr:.2f}, Pixel Sum - {seg.sum():.0f}',
                            save_file=self.log_dir / f'validation/batch/epoch_{epoch}_{idx}.png'
                        )

        # - Clean the seg measures history arrays
        self.model.train_epoch_trgt_seg_msrs = np.array([])
        self.model.train_epoch_pred_seg_msrs = np.array([])
        self.model.train_epoch_outliers = list()

        self.model.val_epoch_trgt_seg_msrs = np.array([])
        self.model.val_epoch_pred_seg_msrs = np.array([])
        self.model.val_epoch_outliers = list()
