import os
import pathlib
import time
import datetime

import numpy as np

from configs.general_configs import CONFIGS_DIR, TRAIN_DATA_FILE
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger
from pytorch import torch_train as tr_train
from tensor_flow import tf_train


if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'{args.model_lib}_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    if args.train:
        print(f'\n== Running train with {args.model_lib} model ==\n')
        if args.model_lib == 'pytorch':
            tr_train.run(
                args=args,
                output_dir=current_run_dir,
                logger=logger
            )
        elif args.model_lib == 'tensor_flow':
            tf_train.run(
                args=args,
                output_dir=current_run_dir,
                logger=logger
            )
    elif args.test:
        print(f'\n== Running test with {args.model_lib} model ==\n')
        data = np.load(str(TRAIN_DATA_FILE), allow_pickle=True)
        if args.model_lib == 'pytorch':
            tr_train.run(
                args=args,
                output_dir=current_run_dir,
                logger=logger
            )
        elif args.model_lib == 'tensor_flow':
            tf_train.run(
                args=args,
                output_dir=current_run_dir,
                logger=logger
            )

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
