import argparse
import os
import pathlib
import time
import datetime

import yaml

from global_configs.general_configs import (
    CONFIGS_DIR,
)
from tensor_flow.utils.tf_utils import train_model, choose_gpu
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    err_log,
    get_data
)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def update_hyper_parameters(hyper_parameters: dict, arguments: argparse.Namespace):
    # - Get hyper-parameter names
    hyp_param_categories = list(hyper_parameters.keys())

    # - Get the argument names
    args = vars(arguments)
    arg_names = list(args.keys())

    # - For each argument
    for arg_name in arg_names:
        for hyp_param_cat in hyp_param_categories:
            # - Get the hyperparameter names fo the category
            hyp_param_names = hyper_parameters.get(hyp_param_cat)

            # - If the argument name is in hyperparameter names for the current category
            if arg_name in hyp_param_names and args.get(arg_name) is not None:
                # - Update it with the relevant value
                hyper_parameters.get(hyp_param_cat)[arg_name] = args.get(arg_name)

    return hyp_params_dict


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
    hyp_params_dict = update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    # - Create the directory for the current run
    if hyp_params_dict.get('training')['load_checkpoint']:
        current_run_dir = pathlib.Path(hyp_params_dict.get('training')['tf_checkpoint_dir']).parent
    else:
        current_run_dir = pathlib.Path(hyp_params_dict.get('general')['output_dir']) / f'train/tensor_flow_{args.name}_{ts}'
        os.makedirs(current_run_dir)

    # - Save the updated hyperparameters to the current run directory
    yaml.dump(
        hyp_params_dict,
        (current_run_dir / 'hyper_params.yml').open(mode='w')
    )

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
    data_dict = get_data(
        data_file=hyp_params_dict.get('data')['train_temp_data_file'],
        data_dir=hyp_params_dict.get('data')['train_data_dir'],
        masks_dir=hyp_params_dict.get('data')['train_mask_dir'],
        logger=logger
    )

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    trained_model = None
    print(f'''
    > Training on {len(data_dict)} examples ...
    ''')
    try:
        trained_model = train_model(
            data_dict=data_dict,
            hyper_parameters=hyp_params_dict,
            output_dir=current_run_dir,
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')

    print(f'''
    ===========================
    == Training on finished! ==
    ===========================
    ''')
    print(f'''
    > Total runtime: {get_runtime(seconds=time.time() - t_start)}
    ''')
