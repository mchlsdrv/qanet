import os
import pathlib
import time
import datetime
import yaml
import matplotlib as mpl
from tensor_flow.utils.tf_utils import train_model, choose_gpu
from clearml import Task
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    err_log,
    update_hyper_parameters, print_pretty_message
)
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
        dir_name = f'{hyp_params_dict.get("general")["experiment_name"]}_{ts}'
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
    trained_model = None

    try:
        trained_model = train_model(
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
        print_pretty_message(message='Testing the SIM+ Data')
        print(f'- Best Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_sim --hyper_params_file train_configs/hyper_params.yml '
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_best_model"]}'
                  )
        print(f'- Last Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_sim --hyper_params_file train_configs/hyper_params.yml '
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_last_model"]}'
                  )

        print_pretty_message(message='Testing the GOWT1 Data')
        print(f'- Best Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_gowt1 --hyper_params_file train_configs/hyper_params.yml'
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_best_model"]}'
                  )
        print(f'- Last Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_gowt1 --hyper_params_file train_configs/hyper_params.yml'
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_last_model"]}'
                  )

        print_pretty_message(message='Testing the HeLa data')
        print(f'- Best Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_hela --hyper_params_file train_configs/hyper_parameters.yml',
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_best_model"]}'
                  )
        print(f'- Last Model:')
        os.system(f'python tf_test.py --architecture {hyp_params_dict.get("model")["architecture"]} '
                  f'--gpu_id {args.gpu_id} --test_hela --hyper_params_file train_configs/hyper_parameters.yml',
                  f'--checkpoint_dir {current_run_dir}/{hyp_params_dict.get("callbacks")["checkpoint_file_last_model"]}'
                  )
