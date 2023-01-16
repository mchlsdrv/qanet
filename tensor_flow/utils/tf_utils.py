import os
from functools import partial
import logging
import logging.config
import threading
import multiprocessing as mlp
import pathlib

import tensorflow as tf
from keras import backend as K

from global_configs.general_configs import (
    LAUNCH_TB,
    METRICS,
    REDUCE_LR_ON_PLATEAU_VERBOSE,
    REDUCE_LR_ON_PLATEAU_MIN_LR,
)
from tensor_flow.configs.general_configs import (
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
)
from utils.aux_funcs import (
    info_log,
    warning_log,
    err_log,
    get_data, str_2_path, print_pretty_message
)
from .tf_data_utils import get_data_loaders, DataLoader
from ..custom.tf_models import (
    RibCage
)

from ..custom.tf_callbacks import (
    ProgressLogCallback
)


class DropBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass


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


def get_callbacks(callback_type: str, hyper_parameters: dict, output_dir: pathlib.Path, logger: logging.Logger = None):
    callbacks = []
    # -------------------
    # Built-in  callbacks
    # -------------------
    tb_prc = None
    if hyper_parameters.get(callback_type)['tensorboard']:
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
                    tensorboard_logs=hyper_parameters.get(callback_type)['tensorboard'],
                    wandb_logs=hyper_parameters.get(callback_type)['wandb'],
                    logger=logger
                )
            )
        # - Launch the tensorboard in a thread
        if TENSOR_BOARD_LAUNCH:
            info_log(logger=logger, message=f'Launching a Tensor Board thread on logdir: \'{output_dir}\'...')
            tb_prc = mlp.Process(
                target=lambda: os.system(f'tensorboard --logdir={output_dir}'),
            )

    if hyper_parameters.get(callback_type)['early_stopping']:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=hyper_parameters.get(callback_type)['early_stopping_monitor'],
                min_delta=hyper_parameters.get(callback_type)['early_stopping_min_delta'],
                patience=hyper_parameters.get(callback_type)['early_stopping_patience'],
                mode=hyper_parameters.get(callback_type)['early_stopping_mode'],
                restore_best_weights=hyper_parameters.get(callback_type)['early_stopping_restore_best_weights'],
                verbose=hyper_parameters.get(callback_type)['early_stopping_verbose'],
            )
        )

    if hyper_parameters.get(callback_type)['terminate_on_nan']:
        callbacks.append(
            tf.keras.callbacks.TerminateOnNaN()
        )

    if hyper_parameters.get(callback_type)['reduce_lr_on_plateau']:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_monitor'],
                factor=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_factor'],
                patience=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_patience'],
                min_delta=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_min_delta'],
                cooldown=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_cooldown'],
                min_lr=REDUCE_LR_ON_PLATEAU_MIN_LR,
                mode=hyper_parameters.get(callback_type)['reduce_lr_on_plateau_mode'],
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


def get_model(mode: str, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    weights_loaded = False

    model_configs = dict(
        input_image_dims=(hyper_parameters.get('augmentations')['crop_height'], hyper_parameters.get('augmentations')['crop_width']),
        drop_block=dict(
            use=hyper_parameters.get('model')['drop_block'],
            keep_prob=hyper_parameters.get('model')['drop_block_keep_prob'],
            block_size=hyper_parameters.get('model')['drop_block_block_size']
        ),
        architecture=hyper_parameters.get('model')['architecture'],
        kernel_regularizer=dict(
            type=hyper_parameters.get('model')['kernel_regularizer_type'],
            l1=hyper_parameters.get('model')['kernel_regularizer_l1'],
            l2=hyper_parameters.get('model')['kernel_regularizer_l2'],
            factor=hyper_parameters.get('model')['kernel_regularizer_factor'],
            mode=hyper_parameters.get('model')['kernel_regularizer_mode']
        ),
        activation=dict(
            type=hyper_parameters.get('model')['activation'],
            max_value=hyper_parameters.get('model')['activation_relu_max_value'],
            negative_slope=hyper_parameters.get('model')['activation_relu_negative_slope'],
            threshold=hyper_parameters.get('model')['activation_relu_threshold'],
            alpha=hyper_parameters.get('model')['activation_leaky_relu_alpha']
        )
    )
    model = RibCage(model_configs=model_configs, output_dir=output_dir, logger=logger)

    checkpoint_dir = str_2_path(path=hyper_parameters.get(mode)['checkpoint_dir'])
    if checkpoint_dir.is_dir():
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
    compilation_configs = dict(
        algorithm=hyper_parameters.get('training')['optimizer'],
        learning_rate=hyper_parameters.get('training')['optimizer_lr'],
        weighted_loss=hyper_parameters.get('training')['weighted_loss'],
        rho=hyper_parameters.get('training')['optimizer_rho'],
        beta_1=hyper_parameters.get('training')['optimizer_beta_1'],
        beta_2=hyper_parameters.get('training')['optimizer_beta_2'],
        amsgrad=hyper_parameters.get('training')['optimizer_amsgrad'],
        momentum=hyper_parameters.get('training')['optimizer_momentum'],
        nesterov=hyper_parameters.get('training')['optimizer_nesterov'],
        centered=hyper_parameters.get('training')['optimizer_centered'],
    )
    model.compile(
        loss=WeightedMSE(weighted=compilation_configs.get('weighted_loss')),
        optimizer=get_optimizer(args=compilation_configs),
        run_eagerly=True,
        metrics=METRICS
    )
    return model, weights_loaded


def get_optimizer(args: dict):
    algorithm = args.get('algorithm')
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
                print_pretty_message(
                    message=f'Running on: {logical_gpus} (GPU #{gpu_id})',
                    delimiter_symbol='='
                )
            elif gpu_id > len(gpus) - 1:
                print_pretty_message(
                    message=f'Running on all GPUs',
                    delimiter_symbol='='
                )
            elif gpu_id < 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                print_pretty_message(
                    message=f'Running on CPU',
                    delimiter_symbol='='
                )
        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)


def train_model(hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    # - Load the data
    data_dict = get_data(mode='training', hyper_parameters=hyper_parameters, logger=logger)

    print_pretty_message(
        message=f'Training on {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # MODEL
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(mode='training', hyper_parameters=hyper_parameters, output_dir=output_dir, logger=logger)

    # - Get the train and the validation data loaders
    train_dl, val_dl = get_data_loaders(mode='training', data_dict=data_dict, hyper_parameters=hyper_parameters, logger=logger)

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(callback_type='training', hyper_parameters=hyper_parameters, output_dir=output_dir, logger=logger)

    # - If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and LAUNCH_TB:
        tb_prc.start()

    # - Train -
    model.fit(
        train_dl,
        batch_size=hyper_parameters.get('training')['batch_size'],
        validation_data=val_dl,
        shuffle=True,
        epochs=hyper_parameters.get('training')['epochs'],
        callbacks=callbacks
    )

    # -> If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and LAUNCH_TB:
        tb_prc.join()


def infer_data(hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    # - Load the data
    data_dict = get_data(mode='inference', hyper_parameters=hyper_parameters, logger=logger)

    inf_dl = DataLoader(
        mode='inference',
        data_dict=data_dict,
        file_keys=list(data_dict.keys()),
        crop_height=hyper_parameters.get('augmentations')['crop_height'],
        crop_width=hyper_parameters.get('augmentations')['crop_width'],
        masks_dir='',
        batch_size=1,
        calculate_seg_measure=False,
        logger=logger
    )

    print_pretty_message(
        message=f'Inferring {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # MODEL
    # -1- Build the model and optionally load the weights
    trained_model, weights_loaded = get_model(mode='inference', hyper_parameters=hyper_parameters, output_dir=output_dir, logger=logger)

    assert weights_loaded, f'Could not load weights from {pathlib.Path(hyper_parameters.get("inference")["checkpoint_dir"])}!'

    # - Infer
    preds = trained_model.infer(data_loader=inf_dl)

    print_pretty_message(
        message=f'Preds: {preds}',
        delimiter_symbol='='
    )
    print_pretty_message(
        message=f'E[Preds]: {preds.mean():.3f}±{preds.std():5f}',
        delimiter_symbol='='
    )


def test_model(model, data_dict, file_tuples, hyper_parameters: dict, output_dir: pathlib.Path, logger: logging.Logger = None):
    # - Get the GT data loader
    test_dl = DataLoader(
        mode='test',
        data_dict=data_dict,
        file_keys=data_dict.keys(),
        crop_height=hyper_parameters.get('training')['crop_height'],
        crop_width=hyper_parameters.get('training')['crop_width'],
        batch_size=hyper_parameters.get('training')['train_batch_size'],
        calculate_seg_measure=hyper_parameters.get('training')['image_height'] > hyper_parameters.get('training')['crop_height'] or hyper_parameters.get('training')['image_width'] > hyper_parameters.get('training')['crop_width'],
        masks_dir=hyper_parameters.get('training')['test_mask_dir'],
        logger=logger
    )

    # -> Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(
        callback_type='test',
        hyper_parameters=hyper_parameters,
        output_dir=output_dir,
        logger=logger
    )

    # -> If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and LAUNCH_TB:
        tb_prc.start()

    # -> Run the test
    print(f'> Testing ...')
    model.evaluate(
        test_dl,
        verbose=1,
        callbacks=callbacks
    )

    # -> If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and LAUNCH_TB:
        tb_prc.join()
