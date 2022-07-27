import os
import datetime
import pathlib

from configs.torch_general_configs import (
    TRAIN_DIR,
    CONFIGS_DIR
)
from utils.torch_aux_funcs import (
    get_device,
    get_logger,
    get_arg_parser, train_model, test_model
)

from configs.torch_general_configs import (
    TEST_GT_DIR,
    TEST_ST_DIR,
)

import warnings

__author__ = 'sidorov@post.bgu.ac.il'

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    # - Configure the GPU to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    trained_model = train_model(
        args=args,
        data_dir=TRAIN_DIR,
        epochs=args.epochs,
        device=device,
        save_dir=current_run_dir,
        logger=logger
    )

    # - TEST -
    # -- GT
    # -*- Get the gold standard test data loader
    if TEST_GT_DIR.is_dir():
        test_model(
            model=trained_model,
            data_dir=TEST_GT_DIR,
            args=args,
            device=device,
            seg_dir_postfix='GT',
            image_prefix='t0',
            seg_prefix='man_seg0',
            save_dir=current_run_dir,
            logger=logger
        )
    # -- ST
    # -*- Get the silver standard test data loader
    if TEST_ST_DIR.is_dir():
        test_model(
            model=trained_model,
            data_dir=TEST_ST_DIR,
            args=args,
            device=device,
            seg_dir_postfix='ST',
            image_prefix='t0',
            seg_prefix='man_seg0',
            save_dir=current_run_dir,
            logger=logger
        )
