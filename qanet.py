import os
import sys
import datetime
import pathlib
import tensorflow as tf
import multiprocessing as mlp
from utils.aux_funcs import (
    info_log,
    choose_gpu,
    get_logger,
    get_model,
    get_arg_parser,
    get_callbacks,
)
from utils.data_utils import (
    # DataLoader,
    # get_data_files,
    get_data_loaders,
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

    # - Get the model
    model, weights_loaded = get_model(
        input_image_dims=(args.crop_size, args.crop_size),
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # - Compile the model
    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,  # (learning_rate=args.learning_rate),
        run_eagerly=True,
        metrics=METRICS

    )

    # - Data loading processes
    main_data_loading_prcs = side_data_loading_prcs = None

    # - Chose the procedure
    if args.inference:
        procedure_name = 'inference'
    elif args.test:
        procedure_name = 'test'
    else:
        procedure_name = 'train'

    # -1- If we want to infer results from the current model
    if args.inference and weights_loaded:
        # - Get the files
        # inference_fls, _ = get_data_files(
        #     data_dir=args.inference_image_dir if args.data_from_single_dir else args.inference_dir,
        #     segmentations_dir=args.inference_seg_dir if args.data_from_single_dir else None,
        #     metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
        #     validation_proportion=0.,
        #     logger=logger
        # )
        #
        # # - Create the DataLoader object
        # inference_dl = DataLoader(
        #     name='INFERENCE',
        #     data_files=inference_fls,
        #     batch_size=args.batch_size,
        #     reload_data=args.reload_data,
        #     logger=logger
        # )
        #

        infer_data_dir = args.inference_image_dir if args.data_from_single_dir else args.inference_dir
        infer_seg_dir = args.inference_seg_dir if args.data_from_single_dir else None

        if isinstance(logger, logging.Logger):
            logger.info(f'- Inferring the images at {infer_data_dir}')

        infer_dl, _ = get_data_loaders(
            main_name=procedure_name,
            side_name='',
            data_dir=infer_data_dir,  # args.inference_image_dir if args.data_from_single_dir else args.inference_dir,
            segmentations_dir=infer_seg_dir,  # args.inference_seg_dir if args.data_from_single_dir else None,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=0.,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )

        # -> Start the train data loading process
        main_data_loading_prcs = mlp.Process(target=infer_dl.enqueue_batches, args=())
        main_data_loading_prcs.start()

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type=procedure_name,
            output_dir=current_run_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None:
            tb_prc.start()

        # - Inference -
        preds = model.predict(
            infer_dl,
            verbose=1,
            callbacks=callbacks
        )
        print(f'''
        PREDICTIONS:
            mean: {preds.mean:.2f} +/- {preds.std:.4f}
        ''')

    # -2- If we want to train a new model
    elif not args.test and not args.inference:
        # - Train model
        train_data_dir = args.train_image_dir if args.data_from_single_dir else args.train_dir,
        train_seg_dir = args.train_seg_dir if args.data_from_single_dir else None
        if isinstance(logger, logging.Logger):
            logger.info(f'- Training the \'RibCage\' model on data from {train_data_dir}')
        # - Get the train and the validation file names, where the split will be determined
        # by the VALIDATION_PROPORTION variable from the configs.general_configs module
        # train_fls, val_fls = get_data_files(
        #     data_dir=args.train_image_dir if args.data_from_single_dir else args.train_dir,
        #     segmentations_dir=args.train_seg_dir if args.data_from_single_dir else None,
        #     metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
        #     validation_proportion=args.validation_proportion,
        #     logger=logger
        # )
        #
        # # - Create the train data loader
        # train_dl = DataLoader(
        #     name='TRAIN',
        #     data_files=train_fls,
        #     batch_size=args.batch_size,
        #     reload_data=args.reload_data,
        #     logger=logger
        # )
        #
        # # -> Start the train data loading process
        # train_data_loading_prcs = mlp.Process(target=train_dl.enqueue_batches, args=())
        # train_data_loading_prcs.start()
        #
        # # - Create the validation data loader
        # val_dl = DataLoader(
        #     name='VALIDATION',
        #     data_files=val_fls,
        #     batch_size=args.batch_size,
        #     reload_data=args.reload_data,
        #     logger=logger
        # )
        # # -> Start the validation data loading process
        # val_data_loading_prcs = mlp.Process(target=val_dl.enqueue_batches, args=())
        # val_data_loading_prcs.start()
        #
        train_dl, val_dl = get_data_loaders(
            main_name=procedure_name,
            side_name='val',
            data_dir=train_data_dir,  # args.train_image_dir if args.data_from_single_dir else args.train_dir,
            segmentations_dir=train_seg_dir,  # args.train_seg_dir if args.data_from_single_dir else None,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=args.validation_proportion,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )

        # -> Start the train data loading process
        main_data_loading_prcs = mlp.Process(target=train_dl.enqueue_batches, args=())
        main_data_loading_prcs.start()

        # -> Start the validation data loading process
        side_data_loading_prcs = mlp.Process(target=val_dl.enqueue_batches, args=())
        side_data_loading_prcs.start()

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type=procedure_name,
            output_dir=current_run_dir,
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

    # -2- If we want to test the current model
    elif args.test and weights_loaded:
        # - Get the files
        # TODO: create a function that returns the DataLoader
        # test_fls, val_fls = get_data_files(
        #     data_dir=args.test_image_dir if args.data_from_single_dir else args.test_dir,
        #     segmentations_dir=args.test_seg_dir if args.data_from_single_dir else None,
        #     metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
        #     validation_proportion=args.validation_proportion,
        #     logger=logger
        # )
        #
        # # - Create the DataLoader object
        # test_dl = DataLoader(
        #     name='TEST',
        #     data_files=test_fls,
        #     batch_size=args.batch_size,
        #     reload_data=args.reload_data,
        #     logger=logger
        # )

        data_dir = args.test_image_dir if args.data_from_single_dir else args.test_dir
        seg_dir = args.test_seg_dir if args.data_from_single_dir else None

        if isinstance(logger, logging.Logger):
            logger.info(f' - Testing the images at {data_dir}')

        test_dl, _ = get_data_loaders(
            main_name=procedure_name,
            side_name='',
            data_dir=data_dir,  # args.test_image_dir if args.data_from_single_dir else args.test_dir,
            segmentations_dir=seg_dir,  # args.test_seg_dir if args.data_from_single_dir else None,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=args.validation_proportion,
            batch_size=args.batch_size,
            reload_data=args.reload_data,
            logger=logger
        )
        # -> Start the data loading process
        main_data_loading_prcs = mlp.Process(target=test_dl.enqueue_batches, args=())
        main_data_loading_prcs.start()

        # - Get the callbacks and optionally the thread which runs the tensorboard
        callbacks, tb_prc = get_callbacks(
            callback_type=procedure_name,
            output_dir=current_run_dir,
            logger=logger
        )

        # - If the setting is to launch the tensorboard process automatically
        if tb_prc is not None:
            tb_prc.start()

        # - Test -
        model.evaluate(
            test_dl,
            verbose=1,
            callbacks=callbacks
        )

    else:
        logger.err_log(f'Could not run the {procedure_name} because the model does not exist!')
        sys.exit(1)

    # - After the training - stop the batch processes for the main and side data loaders
    info_log(logger=logger, message='Joining the main data loading process...')
    main_data_loading_prcs.join()
    info_log(logger=logger, message='The main data loading process was successfully joined!')

    # - Join the side data loading process, is there is one
    if side_data_loading_prcs is not None:
        info_log(logger=logger, message='Joining the side data loading process...')
        side_data_loading_prcs.join()
        info_log(logger=logger, message='The side process was successfully joined!')

    # - If we started the tensorboard thread - stop it after the run ends
    if tb_prc is not None:
        info_log(logger=logger, message='Joining the tensorboard process...')
        tb_prc.join()
        info_log(logger=logger, message='The tensorboard process was successfully joined!')
