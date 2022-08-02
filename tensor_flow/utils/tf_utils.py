import os
import io
from functools import partial
import numpy as np
import logging
import logging.config
import threading
import multiprocessing as mlp
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import augs
from configs.general_configs import TF_LOSS, METRICS, TENSOR_BOARD, TENSOR_BOARD_WRITE_IMAGES, TENSOR_BOARD_WRITE_STEPS_PER_SECOND, TENSOR_BOARD_UPDATE_FREQ, PROGRESS_LOG, SCATTER_PLOT_FIGSIZE, PROGRESS_LOG_INTERVAL, TENSOR_BOARD_LAUNCH, EARLY_STOPPING, EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MODE, EARLY_STOPPING_RESTORE_BEST_WEIGHTS, EARLY_STOPPING_VERBOSE, TERMINATE_ON_NAN, REDUCE_LR_ON_PLATEAU, REDUCE_LR_ON_PLATEAU_MONITOR, \
    REDUCE_LR_ON_PLATEAU_FACTOR, REDUCE_LR_ON_PLATEAU_PATIENCE, REDUCE_LR_ON_PLATEAU_MIN_DELTA, REDUCE_LR_ON_PLATEAU_COOLDOWN, REDUCE_LR_ON_PLATEAU_MIN_LR, REDUCE_LR_ON_PLATEAU_MODE, REDUCE_LR_ON_PLATEAU_VERBOSE, CHECKPOINT, TF_CHECKPOINT_FILE_BEST_MODEL, CHECKPOINT_MONITOR, CHECKPOINT_SAVE_FREQ, CHECKPOINT_SAVE_WEIGHTS_ONLY, CHECKPOINT_MODE, CHECKPOINT_SAVE_BEST_ONLY, CHECKPOINT_VERBOSE, VAL_PROP
from utils.aux_funcs import check_file, info_log, plot_scatter
from . tf_data_utils import get_data_loaders
from .. custom.tf_models import (
    RibCage
)

from .. custom.callbacks import (
    ProgressLogCallback
)


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


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
                    log_type=callback_type,
                    figsize=SCATTER_PLOT_FIGSIZE,
                    log_dir=output_dir,
                    log_interval=PROGRESS_LOG_INTERVAL,
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
                filepath=output_dir / TF_CHECKPOINT_FILE_BEST_MODEL,
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


def get_model(model_configs, checkpoint_dir: pathlib.Path = None, wandb_callback: bool = False, logger: logging.Logger = None):
    weights_loaded = False

    model = RibCage(model_configs=model_configs, logger=logger)

    if checkpoint_dir.is_dir:
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_cpt is not None:
                model.load_weights(latest_cpt)
                weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                logger.exception(f'Can\'t load weighs from \'{checkpoint_dir}\' due to error: {err}')
        else:
            if isinstance(logger, logging.Logger):
                if latest_cpt is not None:
                    logger.info(f'Weights from \'{checkpoint_dir}\' were loaded successfully to the \'RibCage\' model!')
                else:
                    logger.info(f'No weights were found to load in \'{checkpoint_dir}\'!')
    if isinstance(logger, logging.Logger):
        logger.info(model.summary())

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
            if gpu_id > -1:
                tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on: {logical_gpus} (GPU #{gpu_id})
                ====================================================
                ''')
            else:
                if isinstance(logger, logging.Logger):
                    logger.info(f'''
                ====================================================
                > Running on all available devices
                ====================================================
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
                    figure=plot_scatter(
                        x=data.get('Scatter')['x'],
                        y=data.get('Scatter')['y'],
                        save_file=None
                    )
                ),
                step=step
            )


def train_model(args, output_dir: pathlib.Path, logger: logging.Logger = None):
    if check_file(file_path=args.train_data_file):
        # - Load the data
        data = np.load(str(args.train_data_file), allow_pickle=True)

        # MODEL
        # -1- Build the model and optionally load the weights
        model, weights_loaded = get_model(
            model_configs=dict(
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
            checkpoint_dir=pathlib.Path(args.checkpoint_dir),
            logger=logger
        )

        # -2- Compile the model
        model.compile(
            loss=TF_LOSS,
            optimizer=get_optimizer(
                algorithm=args.optimizer,
                args=dict(
                    learning_rate=args.optimizer_lr,
                    rho=args.optimizer_rho,
                    beta_1=args.optimizer_beta_1,
                    beta_2=args.optimizer_beta_2,
                    amsgrad=args.optimizer_amsgrad,
                    momentum=args.optimizer_momentum,
                    nesterov=args.optimizer_nesterov,
                    centered=args.optimizer_centered,
                )
            ),
            run_eagerly=True,
            metrics=METRICS
        )

        # - Get the train and the validation data loaders
        train_dl, val_dl, _ = get_data_loaders(
            data=data,
            train_batch_size=args.batch_size,
            val_prop=VAL_PROP,
            train_augs=augs.train_augs,
            val_augs=augs.val_augs,
            test_augs=None,
            logger=logger
        )

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type='train',
            output_dir=output_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None:
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


def test_model(model, data_file, output_dir: pathlib.Path, logger: logging.Logger = None):
    # -  Load the test data
    data = np.load(str(data_file), allow_pickle=True)

    # - Get the GT data loader
    _, _, test_dl = get_data_loaders(
        data=data,
        train_batch_size=0,
        val_prop=VAL_PROP,
        train_augs=None,
        val_augs=None,
        test_augs=augs.test_augs,
        logger=logger
    )

    # -> Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(
        callback_type='test',
        output_dir=output_dir,
        logger=logger
    )

    # -> If the setting is to launch the tensorboard process automatically
    if tb_prc is not None:
        tb_prc.start()

    # -> Run the test
    print(f'Evaluating')
    model.evaluate(
        test_dl,
        verbose=1,
        callbacks=callbacks
    )
