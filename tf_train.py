import os
import pathlib
import time
import datetime
import numpy as np
from global_configs.general_configs import CONFIGS_DIR, SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX, TRAIN_DATA_ROOT_DIR, SEG_SUB_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger, scan_files, load_images_from_tuple_list
from tensor_flow import train


if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'train/tensor_flow_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    =========================================
    == Running train with TensorFlow model ==
    =========================================
    ''')

    # - Load the data
    fl_tupls = scan_files(
        root_dir=TRAIN_DATA_ROOT_DIR,
        seg_dir_postfix=SEG_DIR_POSTFIX,
        image_prefix=IMAGE_PREFIX,
        seg_prefix=SEG_PREFIX,
        seg_sub_dir=SEG_SUB_DIR
    )

    np.random.shuffle(fl_tupls)

    data_tuples = load_images_from_tuple_list(data_file_tuples=fl_tupls)

    # - Train the model
    train.run(
        data_tuples=data_tuples,
        args=args,
        output_dir=current_run_dir,
        logger=logger
    )

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
