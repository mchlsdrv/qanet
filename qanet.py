import os
import sys
import datetime
import pathlib
import tensorflow as tf
import multiprocessing as mlp
from utils.aux_funcs import (
    info_log,
    err_log,
    choose_gpu,
    get_logger,
    get_model,
    get_arg_parser,
    get_callbacks,
)
from utils.data_utils import (
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

    # MODEL
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(
        input_image_dims=(args.crop_size, args.crop_size),
        checkpoint_dir=pathlib.Path(args.checkpoint_dir),
        logger=logger
    )

    # -2- Compile the model
    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,  # (learning_rate=args.learning_rate),
        run_eagerly=True,
        metrics=METRICS

    )

    # - Data loading processes
    main_data_loading_prcs = side_data_loading_prcs = None

    # - Chose the procedure name
    if args.inference:
        procedure_name = 'inference'
    elif args.test:
        procedure_name = 'test'
    else:
        procedure_name = 'train'

    # PROCEDURE
    # -1- If we want to infer results from the current model
    print(f'weights_loaded: {weights_loaded}')
    if args.inference and weights_loaded:

        # - Get the directory where the inference data is located at
        infer_data_dir = args.inference_image_dir if args.data_from_single_dir else args.inference_dir
        infer_seg_dir = args.inference_seg_dir if args.data_from_single_dir else None

        if isinstance(logger, logging.Logger):
            logger.info(f'- Inferring the images at {infer_data_dir}')

        # - Get the inference data loader
        infer_dl, _ = get_data_loaders(
            main_name=procedure_name,
            side_name='',
            data_dir=infer_data_dir,
            segmentations_dir=infer_seg_dir,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=0.,
            batch_size=2,
            crop_images=True,
            augment_images=False,
            reload_data=args.reload_data,
            logger=logger
        )

        # -> Start the train data loading process
        main_data_loading_prcs = mlp.Process(target=infer_dl.enqueue_batches, args=())
        main_data_loading_prcs.start()

        tb_prc = None

        # - Inference -
        preds = model.infer(
            infer_dl
        )
        print(preds)
        print(f'''
        PREDICTIONS:
            mean: {preds.mean():.2f} +/- {preds.std():.4f}
        ''')

    # -2- If we want to train a new model
    elif not args.test and not args.inference:

        # - Get the directory where the train data is located at
        train_data_dir = args.train_image_dir if args.data_from_single_dir else args.train_dir
        train_seg_dir = args.train_seg_dir if args.data_from_single_dir else None

        if isinstance(logger, logging.Logger):
            logger.info(f'- Training the \'RibCage\' model on data from {train_data_dir}')

        # - Get the train and the validation data loaders
        train_dl, val_dl = get_data_loaders(
            main_name=procedure_name,
            side_name='val',
            data_dir=train_data_dir,
            segmentations_dir=train_seg_dir,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=args.validation_proportion,
            batch_size=args.batch_size,
            crop_images=True,
            augment_images=True,
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

        # - Get the directory where the test data is located at
        data_dir = args.test_image_dir if args.data_from_single_dir else args.test_dir
        seg_dir = args.test_seg_dir if args.data_from_single_dir else None

        if isinstance(logger, logging.Logger):
            logger.info(f' - Testing the images at {data_dir}')

        # - Get the test data loader
        test_dl, _ = get_data_loaders(
            main_name=procedure_name,
            side_name='',
            data_dir=data_dir,
            segmentations_dir=seg_dir,
            metadata_files_regex=None if args.data_from_single_dir else METADATA_FILES_REGEX,
            split_proportion=args.validation_proportion,
            batch_size=args.batch_size,
            crop_images=True,
            augment_images=True,
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
        err_log(logger=logger, message=f'Could not run the {procedure_name} because the model does not exist!')
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

    sys.exit(0)