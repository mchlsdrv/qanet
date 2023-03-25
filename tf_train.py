import os
import pathlib
import time
import datetime
from copy import deepcopy

import yaml
import matplotlib as mpl
from tensor_flow.utils.tf_utils import train_model, choose_gpu, load_checkpoint
from tf_infer import run_inference, infer_all
# from clearml import Task
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    err_log,
    update_hyper_parameters, print_pretty_message
)
from tf_test import run_test
import wandb

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    if not hyp_params_dict.get('general')['debug']:
        mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by the wrong thread" exception

    # - Create the directory for the current run
    if hyp_params_dict.get('training')['load_checkpoint']:
        current_run_dir = pathlib.Path(hyp_params_dict.get('training')['tf_checkpoint_dir']).parent
        dir_name = current_run_dir.name
    else:
        crp_w = hyp_params_dict.get('augmentations')['crop_width']
        crp_h = hyp_params_dict.get('augmentations')['crop_height']
        dir_name = f'{hyp_params_dict.get("general")["name"]}_{ts}'
        current_run_dir = pathlib.Path(
            hyp_params_dict.get('general')['output_dir']) / f'train/tensorflow/{crp_h}x{crp_w}/{dir_name}'
        os.makedirs(current_run_dir)

    # - ClearML
    # task = Task.init(project_name=args.project_name, task_name=dir_name)
    # if not args.local_execution:
    #     task.execute_remotely(args.queue_name)

    # - Save the updated hyperparameters to the current run directory
    yaml.dump(
        hyp_params_dict,
        (current_run_dir / 'hyper_params.yml').open(mode='w')
    )

    # - Configure the logger
    logger = get_logger(
        configs_file=pathlib.Path(hyp_params_dict.get(
            'general')['configs_dir']) / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    =========================================
    == Running train with TensorFlow model ==
    =========================================
    ''')

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    if hyp_params_dict.get('callbacks')['wandb']:
        wandb.init(project=hyp_params_dict.get(
            'callbacks')['wandb_project_name'])

    # - Train model
    model = None

    try:
        model = train_model(
            hyper_parameters=hyp_params_dict,
            output_dir=current_run_dir,
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')

    # - Save the updated hyperparameters to the current run directory
    yaml.dump(
        hyp_params_dict,
        (current_run_dir / 'hyper_params.yml').open(mode='w')
    )
    print(f'''
    ===========================
    == Training on finished! ==
    ===========================
    ''')
    print(f'''
    > Total runtime: {get_runtime(seconds=time.time() - t_start)}
    ''')

    if args.run_tests:
        hyp_params_dict.get('test')['name'] = hyp_params_dict.get('general')['name'] + '_test'

        ckpt_bst = current_run_dir / hyp_params_dict.get("callbacks")["checkpoint_file_best_model"]
        ckpt_lst = current_run_dir / hyp_params_dict.get("callbacks")["checkpoint_file_last_model"]

        hyp_params_dict.get('test')['checkpoint_file_best'] = ckpt_bst
        hyp_params_dict.get('test')['checkpoint_file_last'] = ckpt_lst

        hyp_params_dict.get('test')['gpu_id'] = args.gpu_id

        hyp_params_dict.get('test')['output_dir'] = current_run_dir

        # - SIM+
        test_hyp_params_dict = deepcopy(hyp_params_dict)
        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_bst)
        if ckpt_loaded:
            print_pretty_message(message='Testing the SIM+ Data')
            print(f'- Best Model:')
            test_hyp_params_dict.get('test')['test_sim'] = True
            test_hyp_params_dict.get('test')['type'] = 'best'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_lst)
        if ckpt_loaded:
            print(f'- Last Model:')
            test_hyp_params_dict.get('test')['type'] = 'last'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

        # - GOWT1
        test_hyp_params_dict = deepcopy(hyp_params_dict)
        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_bst)
        if ckpt_loaded:
            print_pretty_message(message='Testing the GOWT1 Data')
            print(f'- Best Model:')
            test_hyp_params_dict.get('test')['test_gowt1'] = True
            test_hyp_params_dict.get('test')['type'] = 'best'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_lst)
        if ckpt_loaded:
            print(f'- Last Model:')
            test_hyp_params_dict.get('test')['type'] = 'last'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

        # - HeLa
        test_hyp_params_dict = deepcopy(hyp_params_dict)
        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_bst)
        if ckpt_loaded:
            print_pretty_message(message='Testing the HeLa Data')
            print(f'- Best Model:')
            test_hyp_params_dict.get('test')['test_hela'] = True
            test_hyp_params_dict.get('test')['type'] = 'best'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_lst)
        if ckpt_loaded:
            print(f'- Last Model:')
            test_hyp_params_dict.get('test')['type'] = 'last'
            run_test(model=model, hyper_parameters=test_hyp_params_dict)

    if args.run_inferences:
        hyp_params_dict.get('inference')['name'] = \
            hyp_params_dict.get('general')['name'] + '_inference'

        ckpt_bst = current_run_dir / hyp_params_dict.get("callbacks")["checkpoint_file_best_model"]
        ckpt_lst = current_run_dir / hyp_params_dict.get("callbacks")["checkpoint_file_last_model"]

        hyp_params_dict.get('inference')['checkpoint_file_best'] = ckpt_bst
        hyp_params_dict.get('inference')['checkpoint_file_last'] = ckpt_lst

        hyp_params_dict.get('inference')['gpu_id'] = args.gpu_id

        hyp_params_dict.get('inference')['output_dir'] = current_run_dir

        # - SIM+
        test_hyp_params_dict = deepcopy(hyp_params_dict)
        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_bst)
        if ckpt_loaded:
            print_pretty_message(message='Inferring the SIM+ Data')
            print(f'- Best Model:')
            infer_all(model=model, hyper_parameters=hyp_params_dict)

        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_lst)
        if ckpt_loaded:
            print(f'- Best Model (inference):')
            infer_all(model=model, hyper_parameters=hyp_params_dict)
