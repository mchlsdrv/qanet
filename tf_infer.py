import logging
import os
import pathlib
import time
import datetime
import tensorflow as tf
import numpy as np
import yaml
from tensor_flow.utils.tf_utils import (
    choose_gpu,
    infer_data, get_model, get_optimizer
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message, check_pathable, str_2_path,
)


def run_inference(model, hyper_parameters: dict):
    t_start = time.time()

    pred_seg_score = -1
    err_pct = -1
    model_name = ''
    # - SIM+ Models
    if hyper_parameters.get('inference')['infer_bgu_3_sim']:
        hyper_parameters.get('inference')['model_name'] = 'BGU_3_SIM'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['bgu_3_sim_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['bgu_3_sim_seg_score']
    elif hyper_parameters.get('inference')['infer_cvut_sim']:
        hyper_parameters.get('inference')['model_name'] = 'CVUT_SIM'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['cvut_sim_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['cvut_sim_seg_score']
    elif hyper_parameters.get('inference')['infer_kth_sim']:
        hyper_parameters.get('inference')['model_name'] = 'KTH_SIM'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['kth_sim_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['kth_sim_seg_score']
    elif hyper_parameters.get('inference')['infer_unsw_sim']:
        hyper_parameters.get('inference')['model_name'] = 'UNSW_SIM'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['unsw_sim_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['unsw_sim_seg_score']
    elif hyper_parameters.get('inference')['infer_dkfz_sim']:
        hyper_parameters.get('inference')['model_name'] = 'DKFZ_SIM'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['dkfz_sim_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['dkfz_sim_seg_score']

    # - GWOT1 Models
    elif hyper_parameters.get('inference')['infer_bgu_4_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'BGU_4_GWOT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['bgu_4_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['bgu_4_gowt1_seg_score']
    elif hyper_parameters.get('inference')['infer_bgu_5_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'BGU_5_GWOT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['bgu_5_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['bgu_5_gowt1_seg_score']
    elif hyper_parameters.get('inference')['infer_unsw_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'UNSW_GWOT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['unsw_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['unsw_gowt1_seg_score']

    model_name = hyper_parameters.get('inference')['model_name']
    if model_name:
        current_run_dir = hyper_parameters.get('inference')['output_dir']
        if check_pathable(current_run_dir):
            current_run_dir = str_2_path(path=current_run_dir)
        if not isinstance(current_run_dir, pathlib.Path) or not current_run_dir.is_dir():
            # - Create the directory for the current run
            exp_name = hyper_parameters.get('inference')['experiment_name']
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            current_run_dir = pathlib.Path(
                pathlib.Path(hyper_parameters.get('general')['output_dir']) /
                f'inference/tensor_flow/{model_name}_{exp_name}_{ts}')
            os.makedirs(current_run_dir)

        print_pretty_message(
            message=f'Current run dir was set to: {current_run_dir}',
            delimiter_symbol='='
        )

        # - Save the updated hyperparameters to the current run directory
        yaml.dump(
            hyp_params_dict,
            (current_run_dir / 'inference_hyper_params.yml').open(mode='w')
        )

        # - Configure the logger
        logger = get_logger(
            configs_file=pathlib.Path(
                hyp_params_dict.get(
                    'general')['configs_dir']) / 'logger_configs.yml',
            save_file=current_run_dir / f'logs.log'
        )

        print_pretty_message(
            message=f'Running inference with TensorFlow model from '
                    f'{hyp_params_dict.get("inference")["checkpoint_file"]}',
            delimiter_symbol='='
        )

        # - Configure the GPU to run on
        choose_gpu(gpu_id=args.gpu_id, logger=logger)

        # MODEL
        if model is None:
            # -1- Build the model and optionally load the weights
            model, weights_loaded = get_model(
                mode='inference',
                hyper_parameters=hyper_parameters,
                output_dir=hyper_parameters.get('inference')['output_dir'],
                logger=logger
            )

            chkpt_fl = hyper_parameters.get("inference")["checkpoint_file"]
            assert weights_loaded, f'Could not load weights from {chkpt_fl}!'

            # - Compile the model
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=get_optimizer(args=hyper_parameters),
                run_eagerly=True,
                metrics=hyper_parameters.get('training')['metrics']
            )

        pred_seg_score = infer_data(
            model=model,
            hyper_parameters=hyp_params_dict,
            output_dir=current_run_dir,
            logger=logger
        )

        print_pretty_message(
            message=f'Total runtime: {get_runtime(seconds=time.time() - t_start)}',
            delimiter_symbol='='
        )
        true_seg_score = hyper_parameters.get("inference")['seg_score']
        err_pct = 100 - (100 * pred_seg_score) / true_seg_score
        print(f'''
        ================================================================
         Inference results on {model_name} model: 
            - True: {true_seg_score:.3f}
            - Pred: {pred_seg_score:.3f}
            - Error: {err_pct:.3f}%
        ================================================================
        ''')

    return pred_seg_score, err_pct


def infer_all(model, hyper_parameters: dict, logger: logging.Logger = None):
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(
        mode='inference',
        hyper_parameters=hyper_parameters,
        output_dir=hyper_parameters.get('inference')['output_dir'],
        logger=logger
    )

    chkpt_fl = hyper_parameters.get("inference")["checkpoint_file"]
    assert weights_loaded, f'Could not load weights from {chkpt_fl}!'

    # - Compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=get_optimizer(args=hyper_parameters),
        run_eagerly=True,
        metrics=hyper_parameters.get('training')['metrics']
    )

    results_dict = dict()

    # - SIM+ Models
    hyper_parameters.get('inference')['infer_unsw_sim'] = True
    _, unsw_sim_err_pct = run_inference(model=model, hyper_parameters=hyp_params_dict)
    results_dict['UNSW-AU'] = unsw_sim_err_pct
    hyper_parameters.get('inference')['infer_unsw_sim'] = False

    hyper_parameters.get('inference')['infer_dkfz_sim'] = True
    _, dkfz_sim_err_pct = run_inference(model=model, hyper_parameters=hyp_params_dict)
    results_dict['DKFZ-GE'] = dkfz_sim_err_pct
    hyper_parameters.get('inference')['infer_dkfz_sim'] = False

    # - GOWT1 Models
    hyper_parameters.get('inference')['infer_bgu_4_gowt1'] = True
    _, bgu_4_gowt1_err_pct = run_inference(model=model, hyper_parameters=hyp_params_dict)
    results_dict['BGU-IL(4)'] = bgu_4_gowt1_err_pct
    hyper_parameters.get('inference')['infer_bgu_4_gowt1'] = False

    hyper_parameters.get('inference')['infer_unsw_gowt1'] = True
    _, bgu_5_gowt1_err_pct = run_inference(model=model, hyper_parameters=hyp_params_dict)
    results_dict['BGU-IL(5)'] = bgu_5_gowt1_err_pct
    hyper_parameters.get('inference')['infer_unsw_gowt1'] = False

    results = np.array(list(results_dict.values()))
    models = np.array(list(results_dict.keys()))

    print(f'''
    - {results_dict}
        > Min: {np.nanmin(results)} ({models[np.nanargmin(results)]})
        > Max: {np.nanmax(results)} ({models[np.nanargmax(results)]})
    ''')


if __name__ == '__main__':
    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyperparameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    # result = run_inference(model=None, hyper_parameters=hyp_params_dict)
    infer_all(model=None, hyper_parameters=hyp_params_dict, logger=None)
