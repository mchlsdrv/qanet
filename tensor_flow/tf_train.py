import os
import pathlib

from utils.aux_funcs import err_log
from .utils.tf_utils import (
    choose_gpu,
    train_model, test_model,
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


def run(args, output_dir, logger):
    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    trained_model = None
    try:
        trained_model = train_model(
            args=args,
            output_dir=output_dir,
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')

    # - TEST -
    # -- Custom
    # -*- Get the test data loader
    if trained_model is not None:
        try:
            test_data_file = pathlib.Path(args.test_data_file)
            if test_data_file.is_file():
                print(f'Testing on custom data from {args.test_data_file}...')
                test_model(
                    model=trained_model,
                    data_file=test_data_file,
                    output_dir=output_dir,
                    logger=logger
                )
        except Exception as err:
            err_log(logger=logger, message=f'{err}')

        # -- GT
        # -*- Get the gold standard test data loader
        try:
            test_gt_data_file = pathlib.Path(args.test_gt_data_file)
            if test_gt_data_file.is_file():
                print(f'Testing on custom data from gold standard {args.test_gt_data_file}...')
                test_model(
                    model=trained_model,
                    data_file=test_gt_data_file,
                    output_dir=output_dir,
                    logger=logger
                )
        except Exception as err:
            err_log(logger=logger, message=f'{err}')

        # -- ST
        # -*- Get the silver standard test data loader
        try:
            test_st_data_file = pathlib.Path(args.test_st_data_file)
            if test_st_data_file.is_file():
                print(f'Testing on custom data from silver standard {args.test_st_data_file}...')
                test_model(
                    model=trained_model,
                    data_file=test_st_data_file,
                    output_dir=output_dir,
                    logger=logger
                )
        except Exception as err:
            err_log(logger=logger, message=f'{err}')
