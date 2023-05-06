import os
import pathlib
import time
import datetime

import torch
import yaml
import matplotlib as mpl

from configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    get_device,
    update_hyper_parameters,
    get_model,
    load_checkpoint
)
from utils.model_utils import test_model

if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyper-parameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)
    if not hyp_params_dict.get('general')['debug']:
        mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by the wrong thread" exception

    # - Create the directory for the current run
    if hyp_params_dict.get('training')['load_checkpoint']:
        current_run_dir = pathlib.Path(hyp_params_dict.get('training')['checkpoint_file']).parent
        dir_name = current_run_dir.name
    else:
        crp_w = hyp_params_dict.get('augmentations')['crop_width']
        crp_h = hyp_params_dict.get('augmentations')['crop_height']
        dir_name = f'{hyp_params_dict.get("general")["name"]}_{ts}'
        current_run_dir = pathlib.Path(
            hyp_params_dict.get('general')['output_dir']) / f'pytorch/train/{crp_h}x{crp_w}/{dir_name}'
        os.makedirs(current_run_dir)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    =====================================
    == Running test with PyTorch model ==
    =====================================
    ''')
    # - Configure the GPU to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)

    # - Build the model
    model = get_model(hyper_parameters=hyp_params_dict)
    model = model.to(device)

    chkpt_fl = pathlib.Path(args.tr_checkpoint_file)
    assert chkpt_fl.is_file(), f'Could not load model from \'{chkpt_fl}\' - file does not exist!'
    load_checkpoint(torch.load(chkpt_fl), model)

    # -- GT
    # -*- Get the gold standard test data loader
    test_data_file = pathlib.Path(args.test_data_file)
    if test_data_file.is_file():
        test_model(
            model=model,
            data_file=test_data_file,
            args=args,
            device=device,
            output_dir=current_run_dir,
            logger=logger
        )

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
