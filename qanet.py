import os
import datetime
import pathlib
import tensorflow as tf
import multiprocessing as mlp
# from utils.general_utils.aux_funcs import (
from utils.aux_funcs import (
    info_log,
    choose_gpu,
    get_logger,
    get_model,
    get_arg_parser,
    get_callbacks,
)
# from utils.data_utils.data_funcs import (
from utils.data_utils import (
    DataLoader,
    get_data_files,
)
import logging.config

from configs.general_configs import (
    CONFIGS_DIR_PATH,
    LOSS,
    OPTIMIZER,
    METRICS,
    METADATA_FILES_REGEX,
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

    model, weights_loaded = get_model(
        input_image_dims=(args.crop_size, args.crop_size),
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # - If we want to test the current model
    if args.test_data_dir and weights_loaded:
        # - Get the files
        test_fls, val_fls = get_data_files(
            data_dir=args.test_image_dir if args.data_from_single_dir else args.test_dir,
            segmentations_dir=args.seg_dir if args.data_from_single_dir else None,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            validation_proportion=args.validation_proportion,
            logger=logger
        )
        
        # - Create the DataLoader object
        test_dl = DataLoader(
            name='TEST',
            data_files=test_fls,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )

        # -> Start the train data loading process
        test_data_loading_prcs = mlp.Process(target=test_dl.enqueue_batches, args=())
        test_data_loading_prcs.start()

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            epochs=args.epochs,
            output_dir=current_run_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None:
            tb_prc.start()

        # - Test
        model.evaluate(
            test_dl,
            verbose=1,
            callbacks=callbacks
        )

    # - If we want to infer results from the current model
    elif args.inference_data_dir and weights_loaded:
        pass

    # - If we want to train a new model
    else:
        # - Train model
        if isinstance(logger, logging.Logger):
            logger.info(f'- Training the RibCage model ...')
        # - Get the train and the validation file names, where the split will be determined
        # by the VALIDATION_PROPORTION variable from the configs.general_configs module
        train_fls, val_fls = get_data_files(
            data_dir=args.image_dir if args.data_from_single_dir else args.root_dir,
            segmentations_dir=args.segmentations_dir if args.data_from_single_dir else None,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            validation_proportion=args.validation_proportion,
            logger=logger
        )
        # if not args.data_from_single_dir:
        #     train_fls, val_fls = get_train_val_split(
        #         data=get_files_from_metadata(
        #                 root_dir=args.root_dir,
        #                 metadata_files_regex=METADATA_FILES_REGEX,
        #                 logger=logger
        #             ),
        #             validation_proportion=args.validation_proportion,
        #             logger=logger
        #     )
        # else:
        #     train_fls, val_fls = get_train_val_split(
        #         data=get_files_from_dir(
        #                 images_dir=args.images_dir,
        #                 segmentations_dir=args.segmentations_dir
        #             ),
        #             validation_proportion=args.validation_proportion,
        #             logger=logger
        #     )

        # - Create the train data loader
        train_dl = DataLoader(
            name='TRAIN',
            data_files=train_fls,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )
        # -> Start the train data loading process
        train_data_loading_prcs = mlp.Process(target=train_dl.enqueue_batches, args=())
        train_data_loading_prcs.start()

        # - Create the validation data loader
        val_dl = DataLoader(
            name='VALIDATION',
            data_files=val_fls,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )
        # -> Start the validation data loading process
        val_data_loading_prcs = mlp.Process(target=val_dl.enqueue_batches, args=())
        val_data_loading_prcs.start()

        # - Train procedure
        model.compile(
            loss=LOSS,
            optimizer=OPTIMIZER(learning_rate=args.learning_rate),
            run_eagerly=True,
            metrics=METRICS

        )

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            epochs=args.epochs,
            output_dir=current_run_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None:
            tb_prc.start()

        # - Train
        model.fit(
            train_dl,
            batch_size=args.batch_size,
            validation_data=val_dl,
            shuffle=True,
            epochs=args.epochs,
            callbacks=callbacks
        )

        # - After the training - stop the batch processes for the train and validation data loaders
        info_log(logger=logger, message='Joining the train process...')
        train_data_loading_prcs.join()
        info_log(logger=logger, message='The train process was successfully joined!')

        info_log(logger=logger, message='Joining the validation process...')
        val_data_loading_prcs.join()
        info_log(logger=logger, message='The validation process was successfully joined!')

        # - If we started the tensorboard thread - stop it after the run ends
        if tb_prc is not None:
            info_log(logger=logger, message='Joining the tensorboard process...')
            tb_prc.join()
            info_log(logger=logger, message='The tensorboard process was successfully joined!')
