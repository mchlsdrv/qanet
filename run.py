import os
import pathlib
import time
import datetime

import numpy as np

from configs.general_configs import CONFIGS_DIR, TRAIN_DATA_FILE
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger
from pytorch import train
# from tensor_flow.train import train as tf_train


if __name__ == '__main__':
    t_start = time.time()

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

    if args.train:
        if args.model_lib == 'pytorch':
            train.run(
                args=args,
                save_dir=current_run_dir,
                logger=logger
            )
        else:
            pass
            # tf_train()
    elif args.test:
        data = np.load(str(TRAIN_DATA_FILE), allow_pickle=True)

    print(f'== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==')
