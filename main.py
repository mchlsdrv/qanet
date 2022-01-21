import os
import datetime
import pathlib
import tensorflow as tf
from tensorflow import keras
from utils.general_utils import aux_funcs
from utils.data_utils import data_funcs
from utils.train_utils import train_funcs
import logging.config

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
    parser = aux_funcs.get_arg_parser()
    args = parser.parse_args()

    current_run_dir = pathlib.Path(args.output_dir) / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    logger = aux_funcs.get_logger(
        configs_file=CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    if isinstance(logger, logging.Logger):
        logger.info(tf.config.list_physical_devices('GPU'))

    aux_funcs.choose_gpu(gpu_id=args.gpu_id, logger=logger)

    input_image_shape = (args.crop_size, args.crop_size, 1)

    # - Train model
    if isinstance(logger, logging.Logger):
        logger.info(f'- Training the feature extractor model ...')
    model, weights_loaded = aux_funcs.get_model(
        checkpoint_dir=pathlib.Path(args.feature_extractor_checkpoint_dir),
        logger=logger
    )
    if not args.no_train:
        # - Train data loader
        train_dl = data_funcs.DataLoader(
            data=dict(),
            crop_shape=(256, 256, 1),
            batch_size=32,
            shuffle=True
        )

        # - Validation data loader
        val_dl = data_funcs.DataLoader(
            data=dict(),
            crop_shape=(256, 256, 1),
            batch_size=32,
            shuffle=True
        )

        train_funcs.train_model(
            model=model,
            data=dict(
                train_dataloader=train_dl,
                val_dataloader=val_dl,
            ),
            callback_configs=dict(
                output_dir_path=pathlib.Path(args.output_dir),
                no_reduce_lr_on_plateau=args.no_reduce_lr_on_plateau_feature_extractor
            ),
            compile_configs=dict(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam(learning_rate=args.feature_extractor_optimizer_lr),
                metrics=['accuracy']
            ),
            fit_configs=dict(
                batch_size=args.feature_extractor_batch_size,
                train_epochs=args.feature_extractor_train_epochs,
                train_steps_per_epoch=args.feature_extractor_train_steps_per_epoch,
                validation_steps_proportion=args.feature_extractor_validation_steps_proportion,
                valdation_freq=1,  # [1, 100, 1500, ...] - validate on these epochs
                shuffle=True,
            ),
            general_configs=dict(
                time_stamp=TS
            ),
            logger=logger
        )
