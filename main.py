import os
import datetime
import pathlib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.general_utils import aux_funcs
from utils.algo_utils import algo_funcs
from utils.data_utils import data_funcs
from utils.train_utils import train_funcs
from utils.image_utils import image_funcs, preprocessing_funcs
from losses import clustering_losses
import logging
import logging.config

from augmentations import clustering_augmentations
from configs.general_configs import (
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
LEARNING_RATE = 1e-4

DATA_PATH = 'C:/Users/mchls/Desktop/University/PhD/projects/QANet/data/Fluo-N2DH-SIM+'
if __name__ == '__main__':
    parser = aux_funcs.get_arg_parcer()
    args = parser.parse_args()


    current_run_dir = pathlib.Path(args.output_dir) / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    logger = aux_funcs.get_logger(
        configs_file = CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file = current_run_dir / f'logs.log'
    )

    if isinstance(logger, logging.Logger):
        logger.info(tf.config.list_physical_devices('GPU'))

    aux_funcs.choose_gpu(gpu_id = args.gpu_id, logger=logger)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    # - Train model
    if isinstance(logger, logging.Logger):
        logger.info(f'- Training the feature extractor model ...')
    model, weights_loaded = aux_funcs.get_model(
        checkpoint_dir=pathlib.Path(args.feature_extractor_checkpoint_dir),
        logger=logger
    )
    if not args.no_train:
        train_ds, val_ds = data_funcs.get_dataset_from_tiff(
            data_dir_path=pathlib.Path(args.train_data_dir),
            input_image_shape=input_image_shape,
            batch_size=args.feature_extractor_batch_size,
            validation_split=args.feature_extractor_validation_split
        )
        train_funcs.train_model(
            model=feat_ext_model,
            data=dict(
                train_dataset = train_ds,
                val_dataset = val_ds,
                X_sample = next(iter(val_ds))[0][0]
            ),
            callback_configs=dict(
                output_dir_path=pathlib.Path(args.output_dir),
                no_reduce_lr_on_plateau = args.no_reduce_lr_on_plateau_feature_extractor
            ),
            compile_configs=dict(
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = keras.optimizers.Adam(learning_rate=args.feature_extractor_optimizer_lr),
                metrics = ['accuracy']
            ),
            fit_configs=dict(
                batch_size = args.feature_extractor_batch_size,
                train_epochs = args.feature_extractor_train_epochs,
                train_steps_per_epoch = args.feature_extractor_train_steps_per_epoch,
                validation_steps_proportion = args.feature_extractor_validation_steps_proportion,
                valdation_freq = 1,  # [1, 100, 1500, ...] - validate on these epochs
                shuffle = True,
            ),
            general_configs=dict(
                time_stamp = TS
            ),
            logger=logger
        )
