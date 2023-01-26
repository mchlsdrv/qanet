import os
import pathlib
import time
import datetime

import pandas as pd

import wandb
import yaml

from tensor_flow.utils.tf_utils import (
    choose_gpu,
    infer_data
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message,
)


if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyperparameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    # - Create the directory for the current run
    if hyp_params_dict.get('training')['load_checkpoint']:
        current_run_dir = pathlib.Path(hyp_params_dict.get('training')['tf_checkpoint_dir']).parent
    else:
        current_run_dir = pathlib.Path(hyp_params_dict.get('general')['output_dir']) / f'inference/tensor_flow_{args.name}_{ts}'
        os.makedirs(current_run_dir)

    print_pretty_message(
        message=f'Current run dir was set to: {current_run_dir}',
        delimiter_symbol='='
    )

    # - Save the updated hyperparameters to the current run directory
    yaml.dump(
        hyp_params_dict,
        (current_run_dir / 'hyper_params.yml').open(mode='w')
    )

    # - Configure the logger
    logger = get_logger(
        configs_file=pathlib.Path(hyp_params_dict.get('general')['configs_dir']) / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print_pretty_message(
        message=f'Running inference with TensorFlow model from {hyp_params_dict.get("inference")["checkpoint_dir"]}',
        delimiter_symbol='='
    )

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    res_dict = infer_data(
        hyper_parameters=hyp_params_dict,
        output_dir=current_run_dir,
        logger=logger
    )

    res_df = pd.DataFrame(res_dict)
    print(res_df.head())
    res_df.to_csv(current_run_dir / 'results.csv')#, index=False)

    print_pretty_message(
        message=f'Total runtime: {get_runtime(seconds=time.time() - t_start)}',
        delimiter_symbol='='
    )
