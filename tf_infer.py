import os
import pathlib
import time
import datetime

from configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger
from tensor_flow import infer


if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'inference/tensor_flow_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    =============================================
    == Running inference with TensorFlow model ==
    =============================================
    ''')
    infer.run(
        args=args,
        output_dir=current_run_dir,
        logger=logger
    )

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
