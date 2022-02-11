import os
import datetime
import pathlib
import tensorflow as tf
from tensorflow import keras
from utils.general_utils.aux_funcs import (
    choose_gpu,
    get_file_names,
    get_train_val_split,
)
from utils.data_utils import data_funcs
import logging.config

from configs.general_configs import (
    IMAGES_DIR,
    SEGMENTATIONS_DIR,
    LEARNING_RATE,
    CONFIGS_DIR_PATH,
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
    parser = aux_funcs.get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = aux_funcs.get_logger(
        configs_file=CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    if isinstance(logger, logging.Logger):
        logger.info(tf.config.list_physical_devices('GPU'))

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    if isinstance(logger, logging.Logger):
        logger.info(f'- Training the RibCage model ...')
    model, weights_loaded = aux_funcs.get_model(
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # - If we want to test the current model
    if args.test_data_dir:
        pass

    # - If we want to infere results from the current model
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
            val_prop=args.validation_proportion
        )

        # - Create the train data loader
        train_dl = data_funcs.DataLoader(
            images_files=train_fls[:, 0],
            seg_files=train_fls[:, 1]
        )

        # - Create the validation data loader
        val_dl = data_funcs.DataLoader(
            images_files=val_fls[:, 0],
            seg_files=val_fls[:, 1]
        )

        # - Train procedure
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
            metrics=['acc']
        )

        # 2.4 Fit model
        model.fit(
            train_dl,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=val_dl,
            shuffle=SHUFFLE,
            # callbacks=callbacks
        )
