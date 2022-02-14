import os
import datetime
import pathlib
import tensorflow as tf
from utils.general_utils.aux_funcs import (
    choose_gpu,
    get_logger,
    get_model,
    get_arg_parser,
    get_file_names,
    get_train_val_split,
    get_callbacks,
)
from utils.data_utils.data_funcs import (
    DataLoader
)
import logging.config

from configs.general_configs import (
    CONFIGS_DIR_PATH,
    LOSS,
    OPTIMIZER,
    METRICS
)

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
    input_image_shape = (args.crop_size, args.crop_size, 1)

    # - Train model
    if isinstance(logger, logging.Logger):
        logger.info(f'- Training the RibCage model ...')
    model, weights_loaded = get_model(
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # - If we want to test the current model
    if args.test_data_dir:
        pass

        # - If we want to infer results from the current model
    elif args.inference_data_dir:
        pass

    # - If we want to train a new model
    else:
        # - Get the train and the validation file names, where the split will be determined
        # by the VALIDATION_PROPORTION variable from the configs.general_configs module
        train_fls, val_fls = get_train_val_split(
            data=get_file_names(
                images_dir=args.images_dir,
                segmentations_dir=args.segmentations_dir
            ),
            validation_proportion=args.validation_proportion
        )

        # - Create the train data loader
        train_dl = DataLoader(
            data_files=train_fls
        )

        # - Create the validation data loader
        val_dl = DataLoader(
            data_files=val_fls
        )

        # - Train procedure
        model.compile(
            loss=LOSS,
            optimizer=OPTIMIZER(learning_rate=args.learning_rate),
            metrics=METRICS
        )

        # - Callbacks
        callbacks = get_callbacks(output_dir=current_run_dir)

        # 2.4 Fit model
        model.fit(
            train_dl,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=val_dl,
            shuffle=True,
            callbacks=callbacks
        )
