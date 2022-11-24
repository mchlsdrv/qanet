import os
from functools import partial
import logging
import logging.config
import threading
import multiprocessing as mlp
import pathlib

import numpy as np
import tensorflow as tf
from keras import backend as K

from global_configs.general_configs import (
    METRICS,
    EARLY_STOPPING,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EARLY_STOPPING_VERBOSE,
    REDUCE_LR_ON_PLATEAU,
    REDUCE_LR_ON_PLATEAU_MONITOR,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE,
    REDUCE_LR_ON_PLATEAU_MIN_DELTA,
    REDUCE_LR_ON_PLATEAU_COOLDOWN,
    REDUCE_LR_ON_PLATEAU_MIN_LR,
    REDUCE_LR_ON_PLATEAU_MODE,
    REDUCE_LR_ON_PLATEAU_VERBOSE,
    DATA_ROOT_DIR,
    SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX, TEST_DATA_DIR
)
from tensor_flow.configs.general_configs import (
    TENSOR_BOARD,
    TENSOR_BOARD_WRITE_IMAGES,
    TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
    TENSOR_BOARD_UPDATE_FREQ,
    PROGRESS_LOG,
    TENSOR_BOARD_LAUNCH,
    CHECKPOINT,
    CHECKPOINT_FILE_BEST_MODEL,
    CHECKPOINT_MONITOR,
    CHECKPOINT_SAVE_FREQ,
    CHECKPOINT_SAVE_WEIGHTS_ONLY,
    CHECKPOINT_MODE,
    CHECKPOINT_SAVE_BEST_ONLY,
    CHECKPOINT_VERBOSE,
    TERMINATE_ON_NAN, LOSS,
)
from utils.aux_funcs import (
    check_file,
    info_log,
    warning_log,
    err_log,
    scatter_plot,
    scan_files
)
from .tf_data_utils import get_data_loaders, DataLoader
from ..custom.tf_models import (
    RibCage
)

from ..custom.tf_callbacks import (
    ProgressLogCallback
)

from .tf_data_utils import get_image_from_figure


class WeightedMSE:
    def __init__(self, weighted=False):
        self.weighted = weighted
        self.mse = tf.keras.losses.MeanSquaredError()

    @staticmethod
    def calc_loss_weights(x):
        # - Compute the histogram of the GT seg measures
        x_hist = tf.histogram_fixed_width(x, value_range=[0.0, 1.0], nbins=10)

        # - Replace the places with 0 occurrences with 1 to avoid division by 0
        x_hist = tf.where(tf.equal(x_hist, 0), tf.ones_like(x_hist), x_hist)

        # - Get the weights for each seg measure region based on its occurrence
        x_weights = tf.divide(1, x_hist)

        # - Convert the weights to float32
        x_weights = tf.cast(x_weights, dtype=tf.float32)

        # - Construct the specific weights to multiply the loss by in each range
        loss_weights = tf.ones_like(x, dtype=tf.float32)

        tf.where(tf.greater_equal(x, 0.0) & tf.less(x, 0.1), x_weights[0], loss_weights)
        tf.where(tf.greater_equal(x, 0.1) & tf.less(x, 0.2), x_weights[1], loss_weights)
        tf.where(tf.greater_equal(x, 0.2) & tf.less(x, 0.3), x_weights[2], loss_weights)
        tf.where(tf.greater_equal(x, 0.3) & tf.less(x, 0.4), x_weights[3], loss_weights)
        tf.where(tf.greater_equal(x, 0.4) & tf.less(x, 0.5), x_weights[4], loss_weights)
        tf.where(tf.greater_equal(x, 0.5) & tf.less(x, 0.6), x_weights[5], loss_weights)
        tf.where(tf.greater_equal(x, 0.6) & tf.less(x, 0.7), x_weights[6], loss_weights)
        tf.where(tf.greater_equal(x, 0.7) & tf.less(x, 0.8), x_weights[7], loss_weights)
        tf.where(tf.greater_equal(x, 0.8) & tf.less(x, 0.9), x_weights[8], loss_weights)
        tf.where(tf.greater_equal(x, 0.9) & tf.less(x, 1.0), x_weights[9], loss_weights)

        return loss_weights

    def __call__(self, y_true, y_pred):
        return self.mse(y_true=y_true, y_pred=y_pred, sample_weight=self.calc_loss_weights(x=y_true) if self.weighted else None)


def weighted_mse(true, pred):
    # - Compute the histogram of the GT seg measures
    true_seg_measure_hist = tf.histogram_fixed_width(true, value_range=[0.0, 1.0], nbins=10)

    # - Replace the places with 0 occurrences with 1 to avoid division by 0
    true_seg_measure_hist = tf.where(tf.equal(true_seg_measure_hist, 0), tf.ones_like(true_seg_measure_hist), true_seg_measure_hist)

    # - Get the weights for each seg measure region based on its occurrence
    seg_measure_weights = tf.divide(1, true_seg_measure_hist)

    # - Convert the weights to float32
    seg_measure_weights = tf.cast(seg_measure_weights, dtype=tf.float32)

    # - Construct the specific weights to multiply the loss by in each range
    btch_weights = tf.ones_like(true, dtype=tf.float32)

    tf.where(tf.greater_equal(true, 0.0) & tf.less(true, 0.1), seg_measure_weights[0], btch_weights)
    tf.where(tf.greater_equal(true, 0.1) & tf.less(true, 0.2), seg_measure_weights[1], btch_weights)
    tf.where(tf.greater_equal(true, 0.2) & tf.less(true, 0.3), seg_measure_weights[2], btch_weights)
    tf.where(tf.greater_equal(true, 0.3) & tf.less(true, 0.4), seg_measure_weights[3], btch_weights)
    tf.where(tf.greater_equal(true, 0.4) & tf.less(true, 0.5), seg_measure_weights[4], btch_weights)
    tf.where(tf.greater_equal(true, 0.5) & tf.less(true, 0.6), seg_measure_weights[5], btch_weights)
    tf.where(tf.greater_equal(true, 0.6) & tf.less(true, 0.7), seg_measure_weights[6], btch_weights)
    tf.where(tf.greater_equal(true, 0.7) & tf.less(true, 0.8), seg_measure_weights[7], btch_weights)
    tf.where(tf.greater_equal(true, 0.8) & tf.less(true, 0.9), seg_measure_weights[8], btch_weights)
    tf.where(tf.greater_equal(true, 0.9) & tf.less(true, 1.0), seg_measure_weights[9], btch_weights)

    return K.mean(K.sum(btch_weights * K.square(true-pred)))


def get_callbacks(callback_type: str, output_dir: pathlib.Path, logger: logging.Logger = None):
    callbacks = []
    # -------------------
    # Built-in  callbacks
    # -------------------
    tb_prc = None
    if TENSOR_BOARD:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=output_dir,
                write_images=TENSOR_BOARD_WRITE_IMAGES,
                write_steps_per_second=TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
                update_freq=TENSOR_BOARD_UPDATE_FREQ,
            )
        )
        if PROGRESS_LOG:
            callbacks.append(
                ProgressLogCallback(
                    log_dir=output_dir,
                    logger=logger
                )
            )
        # - Launch the tensorboard in a thread
        if TENSOR_BOARD_LAUNCH:
            info_log(logger=logger, message=f'Launching a Tensor Board thread on logdir: \'{output_dir}\'...')
            tb_prc = mlp.Process(
                target=lambda: os.system(f'tensorboard --logdir={output_dir}'),
            )

    if EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=EARLY_STOPPING_MONITOR,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                patience=EARLY_STOPPING_PATIENCE,
                mode=EARLY_STOPPING_MODE,
                restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
                verbose=EARLY_STOPPING_VERBOSE,
            )
        )

    if TERMINATE_ON_NAN:
        callbacks.append(
            tf.keras.callbacks.TerminateOnNaN()
        )

    if REDUCE_LR_ON_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=REDUCE_LR_ON_PLATEAU_MONITOR,
                factor=REDUCE_LR_ON_PLATEAU_FACTOR,
                patience=REDUCE_LR_ON_PLATEAU_PATIENCE,
                min_delta=REDUCE_LR_ON_PLATEAU_MIN_DELTA,
                cooldown=REDUCE_LR_ON_PLATEAU_COOLDOWN,
                min_lr=REDUCE_LR_ON_PLATEAU_MIN_LR,
                mode=REDUCE_LR_ON_PLATEAU_MODE,
                verbose=REDUCE_LR_ON_PLATEAU_VERBOSE,
            )
        )

    if CHECKPOINT:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir / CHECKPOINT_FILE_BEST_MODEL,
                monitor=CHECKPOINT_MONITOR,
                verbose=CHECKPOINT_VERBOSE,
                save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
                mode=CHECKPOINT_MODE,
                save_weights_only=CHECKPOINT_SAVE_WEIGHTS_ONLY,
                save_freq=CHECKPOINT_SAVE_FREQ,
            )
        )

    return callbacks, tb_prc


def launch_tensorboard(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th


def get_model(model_configs, compilation_configs, output_dir: pathlib.Path, checkpoint_dir: pathlib.Path = None, logger: logging.Logger = None):
    weights_loaded = False

    model = RibCage(model_configs=model_configs, output_dir=output_dir, logger=logger)

    if model_configs.get('load_checkpoint') and checkpoint_dir.is_dir():
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_cpt is not None:
                model.load_weights(latest_cpt)
                weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                err_log(logger=logger, message=f'Can\'t load weighs from \'{checkpoint_dir}\' due to error: {err}')
        else:
            if isinstance(logger, logging.Logger):
                if latest_cpt is not None:
                    info_log(logger=logger, message=f'Weights from \'{checkpoint_dir}\' were loaded successfully to the \'RibCage\' model!')
                else:
                    warning_log(logger=logger, message=f'No weights were found to load in \'{checkpoint_dir}\'!')
    if isinstance(logger, logging.Logger):
        info_log(logger=logger, message=model.summary())

    # -2- Compile the model
    model.compile(
        loss=WeightedMSE(weighted=compilation_configs.get('weighted_loss')),
        # loss=LOSS,
        optimizer=get_optimizer(
            algorithm=compilation_configs.get('algorithm'),
            args=dict(
                learning_rate=compilation_configs.get('learning_rate'),
                rho=compilation_configs.get('rho'),
                beta_1=compilation_configs.get('beta_1'),
                beta_2=compilation_configs.get('beta_2'),
                amsgrad=compilation_configs.get('amsgrad'),
                momentum=compilation_configs.get('momentum'),
                nesterov=compilation_configs.get('nesterov'),
                centered=compilation_configs.get('centered'),
            )
        ),
        run_eagerly=True,
        metrics=METRICS
    )
    return model, weights_loaded


def get_optimizer(algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'adam':
        optimizer = partial(
            tf.keras.optimizers.Adam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
            amsgrad=args.get('amsgrad'),
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            tf.keras.optimizers.Nadam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            tf.keras.optimizers.Adamax,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad
    elif algorithm == 'adadelta':
        optimizer = partial(
            tf.keras.optimizers.Adadelta,
            rho=args.get('rho'),
        )
    elif algorithm == 'sgd':
        optimizer = partial(
            tf.keras.optimizers.SGD,
            momentum=args.get('momentum'),
            nesterov=args.get('nesterov'),
        )
    elif algorithm == 'rms_prop':
        optimizer = partial(
            tf.keras.optimizers.RMSprop,
            rho=args.get('rho'),
            momentum=args.get('momentum'),
            centered=args.get('centered'),
        )
    return optimizer(learning_rate=args.get('learning_rate'))


def choose_gpu(gpu_id: int = 0, logger: logging.Logger = None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if -1 < gpu_id < len(gpus):
                tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f'''
    ===================================================================================
    == Running on: {logical_gpus} (GPU #{gpu_id}) ==
    ===================================================================================
                ''')
            elif gpu_id > len(gpus) - 1:
                print(f'''
    ======================================
    ==       Running on all GPUs        ==
    ======================================
                ''')
            elif gpu_id < 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                print(f'''
    ======================================
    ==          Running on CPU          ==
    ======================================
                ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)


def write_scalars_to_tensorboard(writer, data: dict, step: int):
    with writer.as_default():
        with tf.device('/cpu:0'):
            # SCALARS
            # - Write the loss
            tf.summary.scalar(
                'Loss',
                data.get('Loss'),
                step=step
            )


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

            # -> Write the ground truth
            disp_gts = data.get('GroundTruth')
            tf.summary.image(
                'GroundTruth',
                disp_gts,
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
                    figure=scatter_plot(
                        x=data.get('Scatter')['x'],
                        y=data.get('Scatter')['y'],
                        save_file=None
                    )
                ),
                step=step
            )


def train_model(data_tuples: np.ndarray, args, output_dir: pathlib.Path, logger: logging.Logger = None):
    if data_tuples.any():
        # MODEL
        # -1- Build the model and optionally load the weights
        model, weights_loaded = get_model(
            model_configs=dict(
                load_checkpoint=args.load_checkpoint,
                input_image_dims=(args.image_height, args.image_width),
                drop_block=dict(
                    use=args.drop_block,
                    keep_prob=args.drop_block_keep_prob,
                    block_size=args.drop_block_block_size
                ),
                kernel_regularizer=dict(
                    type=args.kernel_regularizer_type,
                    l1=args.kernel_regularizer_l1,
                    l2=args.kernel_regularizer_l2,
                    factor=args.kernel_regularizer_factor,
                    mode=args.kernel_regularizer_mode
                ),
                activation=dict(
                    type=args.activation,
                    max_value=args.activation_relu_max_value,
                    negative_slope=args.activation_relu_negative_slope,
                    threshold=args.activation_relu_threshold,
                    alpha=args.activation_leaky_relu_alpha
                )
            ),
            compilation_configs=dict(
                algorithm=args.optimizer,
                learning_rate=args.optimizer_lr,
                weighted_loss=args.weighted_loss,
                rho=args.optimizer_rho,
                beta_1=args.optimizer_beta_1,
                beta_2=args.optimizer_beta_2,
                amsgrad=args.optimizer_amsgrad,
                momentum=args.optimizer_momentum,
                nesterov=args.optimizer_nesterov,
                centered=args.optimizer_centered,
            ),
            checkpoint_dir=pathlib.Path(args.tf_checkpoint_dir),
            output_dir=output_dir,
            logger=logger
        )

        # - Get the train and the validation data loaders
        train_dl, val_dl = get_data_loaders(
            data_tuples=data_tuples,
            train_batch_size=args.batch_size,
            val_prop=args.val_prop,
            logger=logger
        )

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type='train',
            output_dir=output_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None and args.lunch_tb:
            tb_prc.start()

        # - Train -
        model.fit(
            train_dl,
            batch_size=args.batch_size,
            validation_data=val_dl,
            shuffle=True,
            epochs=args.epochs,
            callbacks=callbacks
        )

        # -> If the setting is to launch the tensorboard process automatically
        if tb_prc is not None and args.lunch_tb:
            tb_prc.join()
    else:
        err_log(logger=logger, message=f'No train data was found!')


def test_model(model, data_file, args, output_dir: pathlib.Path, logger: logging.Logger = None):
    if check_file(file_path=args.test_data_file):
        # -  Load the test data
        fl_tupls = scan_files(
            root_dir=DATA_ROOT_DIR / f'test/{TEST_DATA_DIR}',
            seg_dir_postfix=SEG_DIR_POSTFIX,
            image_prefix=IMAGE_PREFIX,
            seg_prefix=SEG_PREFIX
        )

        # - Get the GT data loader
        test_dl = DataLoader(
            data_tuples=fl_tupls,
            batch_size=1,
            logger=logger
        )

        # -> Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type='test',
            output_dir=output_dir,
            logger=logger
        )

        # -> If the setting is to launch the tensorboard process automatically
        if tb_prc is not None and args.lunch_tb:
            tb_prc.start()

        # -> Run the test
        print(f'> Testing ...')
        model.evaluate(
            test_dl,
            verbose=1,
            callbacks=callbacks
        )

        # -> If the setting is to launch the tensorboard process automatically
        if tb_prc is not None and args.lunch_tb:
            tb_prc.join()
    else:
        err_log(logger=logger, message=f'No test data was found!')
