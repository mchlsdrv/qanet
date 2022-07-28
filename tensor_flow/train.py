import os
import datetime
import pathlib
import tensorflow as tf
import logging.config
import wandb

from utils.aux_funcs import (
    choose_gpu,
    get_logger,
    get_model,
    get_arg_parser,
    get_callbacks, get_optimizer,
)

from utils.data_utils import (
    get_data_loaders
)

from configs.general_configs import (
    CONFIGS_DIR_PATH,
    LOSS,
    METRICS,
    METADATA_FILES_REGEX,
    ORIGINAL_IMAGE_SHAPE,
    ORIGINAL_IMAGE_MIN_VAL,
    ORIGINAL_IMAGE_MAX_VAL,
)

import warnings
warnings.filterwarnings("ignore")

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == '__main__':
    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="QANet", entity="bio-vision-lab")

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    if isinstance(logger, logging.Logger):
        logger.info(tf.config.list_physical_devices('GPU'))

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)
    input_image_shape = (args.image_size, args.image_size, 1)

    # MODEL
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(
        model_configs=dict(
            input_image_dims=(args.image_size, args.image_size),
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
        wandb_callback=args.wandb,
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # -2- Compile the model
    model.compile(
        loss=LOSS,
        optimizer=get_optimizer(
            algorithm=args.optimizer,
            args=dict(
                learning_rate=args.learning_rate,
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
    train_dl, val_dl = get_data_loaders(
        data_dir=args.train_dir,
        metadata_configs=dict(
            regex=METADATA_FILES_REGEX,
            shape=ORIGINAL_IMAGE_SHAPE,
            min_val=ORIGINAL_IMAGE_MIN_VAL,
            max_val=ORIGINAL_IMAGE_MAX_VAL
        ),
        split_proportion=args.validation_proportion,
        batch_size=args.batch_size,
        image_size=args.image_size,
        configs=dict(
            augmentation_configs=dict(
                clahe=dict(
                    clip_limit=args.clahe_clip_limit,
                    tile_grid_size=args.clahe_tile_grid_size,
                )
            )
        ),
        reload_data=args.reload_data,
        logger=logger
    )

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(
        callback_type='train',
        output_dir=current_run_dir,
        logger=logger
    )
    if args.wandb:
        wandb.config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }

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

    # - Get the test data loader
    test_dl, _ = get_data_loaders(
        data_dir=args.test_dir,
        metadata_configs=dict(
            regex=METADATA_FILES_REGEX,
            shape=ORIGINAL_IMAGE_SHAPE,
            min_val=ORIGINAL_IMAGE_MIN_VAL,
            max_val=ORIGINAL_IMAGE_MAX_VAL
        ),
        split_proportion=args.validation_proportion,
        batch_size=args.batch_size,
        image_size=args.image_size,
        configs=dict(
            augmentation_configs=dict(
                clahe=dict(
                    clip_limit=args.clahe_clip_limit,
                    tile_grid_size=args.clahe_tile_grid_size,
                )
            )
        ),
        reload_data=args.reload_data,
        logger=logger
    )

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(
        callback_type='test',
        output_dir=current_run_dir,
        logger=logger
    )

    # - If the setting is to launch the tensorboard process automatically
    if tb_prc is not None:
        tb_prc.start()

    # - Test -
    print(f'Evaluating')
    print(f'Evaluating: {test_dl}')
    model.evaluate(
        test_dl,
        verbose=1,
        callbacks=callbacks
    )
    print(f'Evaluating')
