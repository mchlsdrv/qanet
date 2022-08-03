import os
import pathlib

from utils.aux_funcs import err_log
from .utils.tf_utils import (
    choose_gpu,
    test_model,
    get_model,
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
    weights_loaded = False
    try:
        choose_gpu(gpu_id=args.gpu_id, logger=logger)

        # - Load the trained model
        trained_model, weights_loaded = get_model(
            model_configs=dict(
                input_image_dims=(args.image_height, args.image_width),
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
            checkpoint_dir=pathlib.Path(args.checkpoint_dir),
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')

    # - TEST -
    # -- Custom
    # -*- Get the test data loader
    if weights_loaded:
        try:
            test_data_file = pathlib.Path(args.test_data_file)
            if test_data_file.is_file():
                print(f'Testing on custom data from {args.test_data_file}...')
                test_model(
                    model=trained_model,
                    data_file=test_data_file,
                    args=args,
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
                test_model(
                    model=trained_model,
                    data_file=test_gt_data_file,
                    args=args,
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
                test_model(
                    model=trained_model,
                    data_file=test_st_data_file,
                    args=args,
                    output_dir=output_dir,
                    logger=logger
                )
        except Exception as err:
            err_log(logger=logger, message=f'{err}')
