import os

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import logging
import logging.config
import threading
import multiprocessing as mlp
import pathlib
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from tqdm import tqdm

from utils.aux_funcs import (
    info_log,
    get_data,
    print_pretty_message,
    get_data_dict,
    clear_unnecessary_columns, check_pathable
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
plt.style.use('seaborn')


def optimize_learning_rate(model, data_loader, epochs: int = 10,
                           learning_rate_min: float = 0.0001, learning_rate_max: float = 0.1,
                           plot_file: pathlib.Path or str = None) -> (float, float):
    # - The optimal learning rate is set to infinity in case of an error
    opt_lr = np.inf

    # - Make a list of learning rates for testing inside the bounds
    lrs = np.linspace(learning_rate_min, learning_rate_max, epochs)

    # - Place the mean losses here
    mean_epch_losses = np.array([])

    # - Save the weights of the initial model
    init_weights = model.get_weights()

    print_pretty_message(message='Searching for the optimal learning rate')
    for epch, lr in zip(range(epochs), lrs):
        print(f'- Optimization Epoch: {epch+1}/{epochs} ({100 * epch / epochs:.2f}% done)')

        # - Initialize model's weights
        model.set_weights(init_weights)

        # - Place the epoch losses here
        epch_losses = np.array([])
        pbar = tqdm(data_loader)
        for idx, ((imgs, msks), seg_scrs) in enumerate(pbar):
            model.optimizer.lr.assign(lr)

            # - Compute the loss according to the predictions
            with tf.GradientTape() as tape:
                preds = model([imgs, msks], training=True)
                loss = model.compiled_loss(seg_scrs, preds)

            # - Get the weights to adjust according to the loss calculated
            trainable_vars = model.trainable_variables

            # - Calculate gradients
            gradients = tape.gradient(loss, trainable_vars)

            # - Update weights
            model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # - Append the current loss to the past losses
            epch_losses = np.append(epch_losses, loss.numpy())

        mean_epch_losses = np.append(mean_epch_losses, epch_losses.mean())
        print(f'''
        Stats for Epoch {epch}:
            - lr: {lr}
            - mean loss: {mean_epch_losses[-1]:.4f}
            '''
              )

    # - The optimal learning rate is the one with the lowest loss
    opt_lr_min = lrs[np.argmin(mean_epch_losses)]
    opt_lr_max = lrs[np.argmax(mean_epch_losses)]
    if check_pathable(path=plot_file):
        plt.plot(lrs, mean_epch_losses)
        plt.scatter(opt_lr_min, mean_epch_losses.min(), edgecolors='b')
        plt.scatter(opt_lr_max, mean_epch_losses.max(), edgecolors='r')
        plt.legend()
        plt.savefig(plot_file)

    # - Return the initial model's weights
    model.set_weights(init_weights)
    if opt_lr_max < opt_lr_min:
        opt_lr_max = 5 * opt_lr_min
    return opt_lr_min, opt_lr_max


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

    # - Best checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / hyper_parameters.get('callbacks')['checkpoint_file_best_model'],
            monitor=hyper_parameters.get('callbacks')['checkpoint_monitor'],
            verbose=hyper_parameters.get('callbacks')['checkpoint_verbose'],
            save_best_only=True,
            mode=hyper_parameters.get('callbacks')['checkpoint_mode'],
            save_weights_only=hyper_parameters.get('callbacks')['checkpoint_save_weights_only'],
            save_freq=hyper_parameters.get('callbacks')['checkpoint_save_freq']
        )
    )

    # - Last checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / hyper_parameters.get('callbacks')['checkpoint_file_last_model'],
            monitor=hyper_parameters.get('callbacks')['checkpoint_monitor'],
            verbose=hyper_parameters.get('callbacks')['checkpoint_verbose'],
            mode=hyper_parameters.get('callbacks')['checkpoint_mode'],
            save_weights_only=hyper_parameters.get('callbacks')['checkpoint_save_weights_only'],
            save_freq=hyper_parameters.get('callbacks')['checkpoint_save_freq'],
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


def load_checkpoint(model, checkpoint_file: str or pathlib.Path):
    # ckpt_fl = str_2_path(path=checkpoint_file)

    weights_loaded = False
    try:
        model.load_weights(checkpoint_file)
        weights_loaded = True
        print_pretty_message(message=f'> Checkpoint was loaded from \'{checkpoint_file}\'')
    except Exception as err:
        print_pretty_message(
            message=f'<!> Could not load checkpoint from \'{checkpoint_file}\'',
            delimiter_symbol='!'
        )
        print(err)

    return weights_loaded


def get_model(mode: str, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    weights_loaded = False

    model_configs = dict(
        input_image_dims=(hyper_parameters.get('augmentations')['crop_height'],
                          hyper_parameters.get('augmentations')['crop_width']),
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
    model = RibCage(model_configs=model_configs, output_dir=output_dir, logger=logger)
    ckpt_file = hyper_parameters.get(mode)['checkpoint_file']
    weights_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_file)

    if isinstance(logger, logging.Logger):
        info_log(logger=logger, message=model.summary())

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
            self.lr *= self.lr_rdctn_fctr

            # - Update the reduction point array by discarding the last reduction point
            if len(self.lr_rdctn_pts) > 1:
                self.lr_rdctn_pts = self.lr_rdctn_pts[1:]
            else:
                self.lr_rdctn_pts = np.array([])

        return self.lr

    def get_config(self):
        pass


def get_optimizer(args: dict):
    algorithm = args.get('training')['optimizer']
    optimizer = None
    if algorithm == 'adam':
        optimizer = partial(
            tf.keras.optimizers.Adam,
            beta_1=args.get('training')['optimizer_beta_1'],
            beta_2=args.get('training')['optimizer_beta_2'],
            amsgrad=args.get('training')['optimizer_amsgrad'],
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            tf.keras.optimizers.Nadam,
            beta_1=args.get('training')['optimizer_beta_1'],
            beta_2=args.get('training')['optimizer_beta_2'],
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            tf.keras.optimizers.Adamax,
            beta_1=args.get('training')['optimizer_beta_1'],
            beta_2=args.get('training')['optimizer_beta_2'],
        )
    elif algorithm == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad
    elif algorithm == 'adadelta':
        optimizer = partial(
            tf.keras.optimizers.Adadelta,
            rho=args.get('training')['optimizer_rho'],
        )
    elif algorithm == 'sgd':
        optimizer = partial(
            tf.keras.optimizers.SGD,
            momentum=args.get('training')['optimizer_momentum'],
            nesterov=args.get('training')['optimizer_nesterov'],
        )
    elif algorithm == 'rms_prop':
        optimizer = partial(
            tf.keras.optimizers.RMSprop,
            rho=args.get('rho'),
            momentum=args.get('training')['optimizer_momentum'],
            centered=args.get('training')['optimizer_centered'],
        )

    if args.get('training')['learning_rate_scheduler'] == 'cyclical':
        lr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=args.get('training')['learning_rate_scheduler_cyclical_init_lr'],
            maximal_learning_rate=args.get('training')['learning_rate_scheduler_cyclical_max_lr'],
            scale_fn=lambda x: 1 / (2. ** (x - 1)),
            step_size=args.get('training')['learning_rate_scheduler_cyclical_step_size']
        )
    elif args.get('training')['learning_rate_scheduler'] == 'cosine':
        lr = tf.keras.optimizers.schedules.CosineDecay(
            args.get('training')['learning_rate'],
            decay_steps=args.get('training')['learning_rate_scheduler_decay_steps']
        )
    else:
        lr = args.get('training')['learning_rate']
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
    model, weights_loaded = get_model(
        mode='training',
        hyper_parameters=hyper_parameters,
        output_dir=output_dir, logger=logger
    )

    # - Get the train and the validation data loaders
    train_dl, val_dl = get_data_loaders(mode='training', data_dict=data_dict,
                                        hyper_parameters=hyper_parameters,
                                        logger=logger)

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(callback_type='training',
                                      hyper_parameters=hyper_parameters,
                                      output_dir=output_dir, logger=logger)

    # - If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and hyper_parameters.get('callbacks')['tensorboard_launch']:
        tb_prc.start()

    # - Optimize the learning rate
    if hyper_parameters.get('training')['learning_rate_optimization_epochs'] > 0:
        if hyper_parameters.get('training')['learning_rate_scheduler'] == 'cyclical':
            opt_lr_min, opt_lr_max = optimize_learning_rate(
                model=model,
                data_loader=train_dl,
                learning_rate_min=hyper_parameters.get('training')['learning_rate_scheduler_cyclical_init_lr'],
                learning_rate_max=hyper_parameters.get('training')['learning_rate_scheduler_cyclical_max_lr'],
                epochs=hyper_parameters.get('training')['learning_rate_optimization_epochs'],
                plot_file=output_dir / 'learning_rate_optimization_plot.png'
            )

            # - Assign the optimal learning rate to the initial learning rate in the cyclical lr scheduler
            hyper_parameters.get('training')['learning_rate_scheduler_cyclical_init_lr'] = opt_lr_min
            hyper_parameters.get('training')['learning_rate_scheduler_cyclical_max_lr'] = opt_lr_max

            print(f''''
            Learning Rate Optimization Stats:
                - Optimal initial learning rate: {opt_lr_min}
                - Optimal maximal learning rate: {opt_lr_max}'
            ''')
        else:
            opt_lr, _ = optimize_learning_rate(
                model=model,
                data_loader=train_dl,
                learning_rate_min=hyper_parameters.get('training')['learning_rate_min'],
                learning_rate_max=hyper_parameters.get('training')['learning_rate_max'],
                epochs=hyper_parameters.get('training')['learning_rate_optimization_epochs'],
                plot_file=output_dir / 'learning_rate_optimization_plot.png'
            )

            # - Set the optimal learning rate
            hyper_parameters.get('training')['learning_rate'] = opt_lr

            print(f''''
            Learning Rate Optimization Stats:
                - Optimal learning rate: {opt_lr}')
            ''')

    # - Compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=get_optimizer(args=hyper_parameters),
        run_eagerly=True,
        metrics=hyper_parameters.get('training')['metrics']
    )

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
    if tb_prc is not None and hyper_parameters.get('callbacks')['tensorboard_launch']:
        tb_prc.join()

    return model


def test_model(model, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    test_res_df = None
    df_fl = pathlib.Path(hyper_parameters.get('test')['dataframe_file'])
    if df_fl.is_file():
        # - Load the dataframe
        test_res_df = pd.read_csv(df_fl)

        # - Clear unnecessary columns
        test_res_df = clear_unnecessary_columns(dataframe=test_res_df)

        # - Clear the nans
        test_res_df = test_res_df.loc[~test_res_df.loc[:, 'seg_score'].isna()].reset_index(drop=True)

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
        if model is None:
            model, weights_loaded = get_model(
                mode='test',
                hyper_parameters=hyper_parameters,
                output_dir=output_dir, logger=logger)

            chkpt_dir = hyper_parameters.get("test")["checkpoint_dir"]
            assert weights_loaded, f'Could not load weights from {pathlib.Path(chkpt_dir)}!'

        # - Infer
        pred_df = model.test(data_loader=test_dl)

        test_res_df.loc[test_res_df.loc[:, 'image_file'].isin(pred_df.loc[:, 'image_file']), 'pred_seg_score'] = \
            pred_df.loc[:, 'seg_score']

        print_pretty_message(
            message=f'Testing {len(data_dict)} images'
        )

    return test_res_df


def infer_data(model, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
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

    # - Infer
    prediction = model.infer(data_loader=inf_dl)

    return prediction
