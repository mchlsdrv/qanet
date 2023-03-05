import os
import io
import numpy as np
from functools import partial
import logging
import logging.config
import threading
import multiprocessing as mlp
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K

from utils.aux_funcs import (
    info_log,
    warning_log,
    err_log,
    get_data, str_2_path, print_pretty_message, get_data_dict,
    clear_unnecessary_columns
)

from .tf_data_utils import (
    get_data_loaders,
    DataLoader
)

from ..custom.tf_models import (
    RibCage
)

from ..custom.tf_callbacks import (
    ProgressLogCallback
)

tf.config.run_functions_eagerly(False)


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

        tf.where(tf.greater_equal(x, 0.0) & tf.less(x, 0.1), x_weights[0],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.1) & tf.less(x, 0.2), x_weights[1],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.2) & tf.less(x, 0.3), x_weights[2],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.3) & tf.less(x, 0.4), x_weights[3],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.4) & tf.less(x, 0.5), x_weights[4],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.5) & tf.less(x, 0.6), x_weights[5],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.6) & tf.less(x, 0.7), x_weights[6],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.7) & tf.less(x, 0.8), x_weights[7],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.8) & tf.less(x, 0.9), x_weights[8],
                 loss_weights)
        tf.where(tf.greater_equal(x, 0.9) & tf.less(x, 1.0), x_weights[9],
                 loss_weights)

        return loss_weights

    def __call__(self, y_true, y_pred):
        return self.mse(y_true=y_true, y_pred=y_pred,
                        sample_weight=self.calc_loss_weights(
                            x=y_true) if self.weighted else None)


def weighted_mse(true, pred):
    # - Compute the histogram of the GT seg measures
    true_seg_measure_hist = tf.histogram_fixed_width(true,
                                                     value_range=[0.0, 1.0],
                                                     nbins=10)

    # - Replace the places with 0 occurrences with 1 to avoid division by 0
    true_seg_measure_hist = tf.where(tf.equal(true_seg_measure_hist, 0),
                                     tf.ones_like(true_seg_measure_hist),
                                     true_seg_measure_hist)

    # - Get the weights for each seg measure region based on its occurrence
    seg_measure_weights = tf.divide(1, true_seg_measure_hist)

    # - Convert the weights to float32
    seg_measure_weights = tf.cast(seg_measure_weights, dtype=tf.float32)

    # - Construct the specific weights to multiply the loss by in each range
    btch_weights = tf.ones_like(true, dtype=tf.float32)

    tf.where(tf.greater_equal(true, 0.0) & tf.less(true, 0.1),
             seg_measure_weights[0], btch_weights)
    tf.where(tf.greater_equal(true, 0.1) & tf.less(true, 0.2),
             seg_measure_weights[1], btch_weights)
    tf.where(tf.greater_equal(true, 0.2) & tf.less(true, 0.3),
             seg_measure_weights[2], btch_weights)
    tf.where(tf.greater_equal(true, 0.3) & tf.less(true, 0.4),
             seg_measure_weights[3], btch_weights)
    tf.where(tf.greater_equal(true, 0.4) & tf.less(true, 0.5),
             seg_measure_weights[4], btch_weights)
    tf.where(tf.greater_equal(true, 0.5) & tf.less(true, 0.6),
             seg_measure_weights[5], btch_weights)
    tf.where(tf.greater_equal(true, 0.6) & tf.less(true, 0.7),
             seg_measure_weights[6], btch_weights)
    tf.where(tf.greater_equal(true, 0.7) & tf.less(true, 0.8),
             seg_measure_weights[7], btch_weights)
    tf.where(tf.greater_equal(true, 0.8) & tf.less(true, 0.9),
             seg_measure_weights[8], btch_weights)
    tf.where(tf.greater_equal(true, 0.9) & tf.less(true, 1.0),
             seg_measure_weights[9], btch_weights)

    return K.mean(K.sum(btch_weights * K.square(true - pred)))


def get_callbacks(callback_type: str, hyper_parameters: dict, output_dir: pathlib.Path, logger: logging.Logger = None):
    callbacks = []
    # -------------------
    # Built-in  callbacks
    # -------------------
    tb_prc = None
    if hyper_parameters.get('callbacks')['tensorboard']:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=output_dir,
                write_images=hyper_parameters.get('callbacks')[
                    'tensorboard_write_images'],
                write_steps_per_second=hyper_parameters.get('callbacks')[
                    'tensorboard_write_steps_per_second'],
                update_freq=hyper_parameters.get('callbacks')[
                    'tensorboard_update_freq'],
            )
        )
        if hyper_parameters.get('callbacks')['progress_log']:
            callbacks.append(
                ProgressLogCallback(
                    log_dir=output_dir,
                    tensorboard_logs=hyper_parameters.get('callbacks')[
                        'tensorboard'],
                    wandb_logs=hyper_parameters.get('callbacks')['wandb'],
                    logger=logger
                )
            )
        # - Launch the tensorboard in a thread
        if hyper_parameters.get('callbacks')['tensorboard_launch']:
            info_log(logger=logger,
                     message=f'Launching a Tensor Board thread on logdir: '
                             f'\'{output_dir}\'...')
            tb_prc = mlp.Process(
                target=lambda: os.system(f'tensorboard --logdir={output_dir}'),
            )

    if hyper_parameters.get('callbacks')['early_stopping']:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=hyper_parameters.get('callbacks')[
                    'early_stopping_monitor'],
                min_delta=hyper_parameters.get('callbacks')[
                    'early_stopping_min_delta'],
                patience=hyper_parameters.get('callbacks')[
                    'early_stopping_patience'],
                mode=hyper_parameters.get('callbacks')['early_stopping_mode'],
                restore_best_weights=hyper_parameters.get('callbacks')[
                    'early_stopping_restore_best_weights'],
                verbose=hyper_parameters.get('callbacks')[
                    'early_stopping_verbose'],
            )
        )

    if hyper_parameters.get('callbacks')['terminate_on_nan']:
        callbacks.append(
            tf.keras.callbacks.TerminateOnNaN()
        )

    if hyper_parameters.get('callbacks')['reduce_lr_on_plateau']:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_monitor'],
                factor=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_factor'],
                patience=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_patience'],
                min_delta=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_min_delta'],
                cooldown=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_cooldown'],
                min_lr=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_min_lr'],
                mode=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_mode'],
                verbose=hyper_parameters.get('callbacks')['reduce_lr_on_plateau_verbose'],
            )
        )

    if hyper_parameters.get('callbacks')['checkpoint']:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir / hyper_parameters.get('callbacks')[
                    'checkpoint_file_best_model'],
                monitor=hyper_parameters.get('callbacks')['checkpoint_monitor'],
                verbose=hyper_parameters.get('callbacks')['checkpoint_verbose'],
                save_best_only=hyper_parameters.get('callbacks')[
                    'checkpoint_save_best_only'],
                mode=hyper_parameters.get('callbacks')['checkpoint_mode'],
                save_weights_only=hyper_parameters.get('callbacks')[
                    'checkpoint_save_weights_only'],
                save_freq=hyper_parameters.get('callbacks')[
                    'checkpoint_save_freq'],
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
        input_image_dims=(hyper_parameters.get('augmentations')['crop_height'],
                          hyper_parameters.get('augmentations')['crop_width']),
        drop_block=dict(
            use=hyper_parameters.get('regularization')['drop_block'],
            keep_prob=hyper_parameters.get('regularization')['drop_block_keep_prob'],
            block_size=hyper_parameters.get('regularization')['drop_block_block_size']
        ),
        architecture=hyper_parameters.get('model')['architecture'],
        kernel_regularizer=dict(
            type=hyper_parameters.get('regularization')[
                'kernel_regularizer_type'],
            l1=hyper_parameters.get('regularization')['kernel_regularizer_l1'],
            l2=hyper_parameters.get('regularization')['kernel_regularizer_l2'],
            factor=hyper_parameters.get('regularization')[
                'kernel_regularizer_factor'],
            mode=hyper_parameters.get('regularization')[
                'kernel_regularizer_mode']
        ),
        activation=dict(
            type=hyper_parameters.get('model')['activation'],
            max_value=hyper_parameters.get('model')[
                'activation_relu_max_value'],
            negative_slope=hyper_parameters.get('model')[
                'activation_relu_negative_slope'],
            threshold=hyper_parameters.get('model')[
                'activation_relu_threshold'],
            alpha=hyper_parameters.get('model')['activation_leaky_relu_alpha']
        )
    )
    model = RibCage(model_configs=model_configs, output_dir=output_dir,
                    logger=logger)

    checkpoint_dir = str_2_path(path=hyper_parameters.get(mode)['checkpoint_dir'])
    if checkpoint_dir.is_dir():
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_cpt is not None:
                model.load_weights(latest_cpt)
                weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                err_log(logger=logger,
                        message=f'Can\'t load weighs from '
                                f'\'{checkpoint_dir}\' due to error: {err}')
        else:
            if isinstance(logger, logging.Logger):
                if latest_cpt is not None:
                    info_log(logger=logger,
                             message=f'Weights from '
                                     f'\'{checkpoint_dir}\' were loaded '
                                     f'successfully to the \'RibCage\' model!')
                else:
                    warning_log(logger=logger,
                                message=f'No weights were found to load in '
                                        f'\'{checkpoint_dir}\'!')
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
        cyclical_lr=hyper_parameters.get('callbacks')['cyclical_lr'],
        cyclical_lr_init_lr=hyper_parameters.get('callbacks')['cyclical_lr_init_lr'],
        cyclical_lr_max_lr=hyper_parameters.get('callbacks')['cyclical_lr_max_lr'],
        cyclical_lr_step_size=hyper_parameters.get('callbacks')['cyclical_lr_step_size'],
        lr_reduction_scheduler=hyper_parameters.get('callbacks')['lr_reduction_scheduler'],
        lr_reduction_scheduler_factor=hyper_parameters.get('callbacks')['lr_reduction_scheduler_factor'],
        lr_reduction_scheduler_decay_steps=hyper_parameters.get('callbacks')['lr_reduction_scheduler_decay_steps'],
    )
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=get_optimizer(args=compilation_configs),
        run_eagerly=True,
        metrics=hyper_parameters.get('training')['metrics']
    )
    return model, weights_loaded


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, lr_reduction_points: list, lr_reduction_factor: float = 0.3):
        self.lr = initial_learning_rate
        self.lr_rdctn_pts = np.array(lr_reduction_points, dtype=np.int16)
        self.lr_rdctn_fctr = lr_reduction_factor

    def __call__(self, step):
        # - If there are points at which we want to reduce the learning rate, and the current step is greater than the
        # next number of epochs
        if self.lr_rdctn_pts.any() and step > self.lr_rdctn_pts[0]:

            # - Reduce the learning rate by the factor
            self.lr = self.lr_rdctn_fctr * self.lr

            # - Update the reduction point array by discarding the last reduction point
            if len(self.lr_rdctn_pts) > 1:
                self.lr_rdctn_pts = self.lr_rdctn_pts[1:]
            else:
                self.lr_rdctn_pts = np.array([])

        return self.lr


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

    if args.get('cyclical_lr'):
        lr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=args.get('cyclical_lr_init_lr'),
            maximal_learning_rate=args.get('cyclical_lr_max_lr'),
            scale_fn=lambda x: 1 / (2. ** (x - 1)),
            step_size=args.get('cyclical_lr_step_size')
        )
    elif args.get('lr_reduction_scheduler') == 'cosine':
        lr = tf.keras.optimizers.schedules.CosineDecay(
            args.get('learning_rate'),
            decay_steps=args.get('lr_reduction_scheduler_decay_steps')
        )
    else:
        lr = args.get('learning_rate')
    return optimizer(learning_rate=lr)


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
    hyper_parameters.get('training')['train_data_len'] = len(data_dict)
    print_pretty_message(
        message=f'Training on {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # MODEL
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(mode='training',
                                      hyper_parameters=hyper_parameters,
                                      output_dir=output_dir, logger=logger)

    # - Get the train and the validation data loaders
    train_dl, val_dl = get_data_loaders(mode='training', data_dict=data_dict,
                                        hyper_parameters=hyper_parameters,
                                        logger=logger)

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(callback_type='training',
                                      hyper_parameters=hyper_parameters,
                                      output_dir=output_dir, logger=logger)

    # - If the setting is to launch the tensorboard process automatically
    if tb_prc is not None \
            and hyper_parameters.get('callbacks')['tensorboard_launch']:
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
    if tb_prc is not None and \
            hyper_parameters.get('callbacks')['tensorboard_launch']:
        tb_prc.join()


def infer_data(hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    # - Load the data
    data_dict = get_data(mode='inference', hyper_parameters=hyper_parameters,
                         logger=logger)

    inf_dl = DataLoader(
        mode='inference',
        data_dict=data_dict,
        file_keys=list(data_dict.keys()),
        crop_height=hyper_parameters.get('augmentations')['crop_height'],
        crop_width=hyper_parameters.get('augmentations')['crop_width'],
        batch_size=1,
        logger=logger
    )

    print_pretty_message(
        message=f'Inferring {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # MODEL
    # -1- Build the model and optionally load the weights
    trained_model, weights_loaded = get_model(mode='inference',
                                              hyper_parameters=hyper_parameters,
                                              output_dir=output_dir,
                                              logger=logger)

    chkpt_dir = hyper_parameters.get("inference")["checkpoint_dir"]
    assert weights_loaded, f'Could not load weights from {chkpt_dir}!'

    # - Infer
    preds_dict = trained_model.infer(data_loader=inf_dl)

    results = np.array(list(preds_dict.values()))[:, -1]
    print(f'''
    > Preds: 
        {results}
    '''
          )
    print_pretty_message(
        message=f'E[Preds]: {results.mean():.3f}±{results.std():5f}',
        delimiter_symbol='='
    )

    return preds_dict


def test_model(hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    test_res_df = None
    df_fl = pathlib.Path(hyper_parameters.get('test')['dataframe_file'])
    if df_fl.is_file():
        # - Load the dataframe
        test_res_df = pd.read_csv(df_fl)

        # - Clear unnecessary columns
        test_res_df = clear_unnecessary_columns(dataframe=test_res_df)

        # - Clear the nans
        test_res_df = test_res_df.loc[~test_res_df.loc[:, 'seg_score'].isna()] \
            .reset_index(drop=True)

        # - Create the data tuples for the to be fed into the get_data_dict
        data_file_tuples = [(img_fl, pred_msk_fl) for
                            img_fl, _, pred_msk_fl, _ in test_res_df.values]

        # - Construct the data dictionary containing the image files, images, mask files and masks
        data_dict = get_data_dict(data_file_tuples=data_file_tuples)

        test_dl = DataLoader(
            mode='test',
            data_dict=data_dict,
            file_keys=list(data_dict.keys()),
            crop_height=hyper_parameters.get('augmentations')['crop_height'],
            crop_width=hyper_parameters.get('augmentations')['crop_width'],
            batch_size=1,
            logger=logger
        )

        # MODEL
        # -1- Build the model and optionally load the weights
        trained_model, weights_loaded = get_model(
            mode='test',
            hyper_parameters=hyper_parameters,
            output_dir=output_dir, logger=logger)

        chkpt_dir = hyper_parameters.get("test")["checkpoint_dir"]
        assert weights_loaded, f'Could not load weights from {pathlib.Path(chkpt_dir)}!'

        # - Infer
        pred_df = trained_model.test(data_loader=test_dl)

        test_res_df.loc[test_res_df.loc[:, 'image_file'].isin(pred_df.loc[:, 'image_file']), 'pred_seg_score'] = \
            pred_df.loc[:, 'seg_score']

        print_pretty_message(
            message=f'Testing {len(data_dict)} images'
        )

    return test_res_df
