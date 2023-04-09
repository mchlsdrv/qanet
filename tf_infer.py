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
    infer_data, get_model, get_optimizer, load_checkpoint
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message, info_log,
)


def run_inference(model, hyper_parameters: dict, logger: logging.Logger = None):
    t_start = time.time()

    model_name = hyper_parameters.get("inference")['model_name']
    true_seg_score = hyper_parameters.get("inference")['seg_score']
    pred_seg_score = -1
    err_pct = -1

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

        print_pretty_message(
            message=f'Running inference with TensorFlow model from '
                    f'{hyper_parameters.get("inference")["checkpoint_file"]}',
            delimiter_symbol='='
        )

        # - Compile the model
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=get_optimizer(args=hyper_parameters),
            run_eagerly=True,
            metrics=hyper_parameters.get('training')['metrics']
        )

    pred_seg_score = infer_data(
        model=model,
        hyper_parameters=hyper_parameters,
        output_dir=hyper_parameters.get("inference")["output_dir"],
        logger=logger
    )

    print_pretty_message(
        message=f'Total runtime: {get_runtime(seconds=time.time() - t_start)}',
        delimiter_symbol='='
    )
    err_pct = 100 - (100 * pred_seg_score) / true_seg_score

    return true_seg_score, pred_seg_score, err_pct


def update_inference_hyper_parameters(hyper_parameters: dict):
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
        hyper_parameters.get('inference')['model_name'] = 'BGU_4_GOWT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['bgu_4_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['bgu_4_gowt1_seg_score']
    elif hyper_parameters.get('inference')['infer_bgu_5_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'BGU_5_GOWT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['bgu_5_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['bgu_5_gowt1_seg_score']
    elif hyper_parameters.get('inference')['infer_kth_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'KTH_GOWT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['kth_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['kth_gowt1_seg_score']
    elif hyper_parameters.get('inference')['infer_unsw_gowt1']:
        hyper_parameters.get('inference')['model_name'] = 'UNSW_GOWT1'
        hyper_parameters.get('inference')['data_dir'] = hyper_parameters.get('inference')['unsw_gowt1_data_dir']
        hyper_parameters.get('inference')['seg_score'] = hyper_parameters.get('inference')['unsw_gowt1_seg_score']


def print_results(results: dict, logger: logging.Logger = None):
    # print('results', results)
    mdl_names, true_vals, pred_vals, errs = [], np.array([]), np.array([]), np.array([])
    for mdl_name, val_dict in results.items():
        true = val_dict.get('true')
        pred = val_dict.get('predicted')
        err = val_dict.get('error(%)')

        mdl_names.append(mdl_name)
        true_vals = np.append(true_vals, true)
        pred_vals = np.append(pred_vals, pred)
        errs = np.append(errs, err)

        msg = f'''
    ================================================
    Model - {mdl_name}
        > True: {true:.3f}
        > Predicted: {pred:.3f}
        > Error (%): {err:.3f}%
    ================================================
            '''
        if isinstance(logger, logging.Logger):
            info_log(logger=logger, message=msg)
        else:
            print(msg)

    min_err_idx = np.nanargmin(np.abs(errs))
    max_err_idx = np.nanargmax(np.abs(errs))
    final_res_msg = f'''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FINAL RESULTS @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@
    @@@   > Min error ({mdl_names[min_err_idx]}): {errs[min_err_idx]:.3f}% 
    @@@        - true: {true_vals[min_err_idx]:.3f}
    @@@        - pred: {pred_vals[min_err_idx]:.3f} 
    @@@   > Max error ({mdl_names[max_err_idx]}): {errs[max_err_idx]:.3f}%
    @@@        - true: {true_vals[max_err_idx]:.3f}
    @@@        - pred: {pred_vals[max_err_idx]:.3f} 
    @@@    
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    '''
    if isinstance(logger, logging.Logger):
        info_log(logger=logger, message=final_res_msg)
    else:
        print(final_res_msg)
    return final_res_msg


def infer_all(model, hyper_parameters: dict, logger: logging.Logger = None):
    if model is None:
        # -1- Build the model and optionally load the weights
        model, weights_loaded = get_model(
            mode='inference',
            hyper_parameters=hyper_parameters,
            output_dir=hyper_parameters.get('inference')['output_dir'],
            logger=logger
        )

        # - Compile the model
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=get_optimizer(args=hyper_parameters),
            run_eagerly=True,
            metrics=hyper_parameters.get('training')['metrics']
        )

    results_dict = dict()

    # ---------------
    # - SIM+ Models -
    # ---------------
    print_pretty_message(message=f'> Inferring SIM+ Models', delimiter_symbol='*')

    print(f'\n\t> KTH-SE (SIM+)')
    hyper_parameters.get('inference')['infer_kth_sim'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['KTH-SE (SIM+)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_kth_sim'] = False

    print(f'\n\t> UNSW-AU (SIM+)')
    hyper_parameters.get('inference')['infer_unsw_sim'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['UNSW-AU (SIM+)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_unsw_sim'] = False

    print(f'\n\t> DKFZ-DE (SIM+)')
    hyper_parameters.get('inference')['infer_dkfz_sim'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['DKFZ-GE (SIM+)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_dkfz_sim'] = False

    # ----------------
    # - GOWT1 Models -
    # ----------------
    print_pretty_message(message=f'> Inferring GOWT1 Models', delimiter_symbol='*')

    print(f'\n\t> BGU-IL (4) (GOWT1)')
    hyper_parameters.get('inference')['infer_bgu_4_gowt1'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['BGU-IL(4) (GOWT1)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_bgu_4_gowt1'] = False

    print(f'\n\t> BGU-IL (5) (GOWT1)')
    hyper_parameters.get('inference')['infer_bgu_5_gowt1'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['BGU-IL(5) (GOWT1)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_bgu_5_gowt1'] = False

    print(f'\n\t> KTH-SE (GOWT1)')
    hyper_parameters.get('inference')['infer_kth_gowt1'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['KTH-SE (GOWT1)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_kth_gowt1'] = False

    print(f'\n\t> UNSW-AU (GOWT1)')
    hyper_parameters.get('inference')['infer_unsw_gowt1'] = True
    update_inference_hyper_parameters(hyper_parameters=hyper_parameters)
    true, pred, err_pct = run_inference(model=model, hyper_parameters=hyper_parameters)
    results_dict['UNSW-AU (GOWT1)'] = {
        'true': true,
        'predicted': pred,
        'error(%)': err_pct
    }
    hyper_parameters.get('inference')['infer_unsw_gowt1'] = False

    return results_dict


if __name__ == '__main__':
    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyperparameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    name = args.name
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_run_dir = pathlib.Path(
        pathlib.Path(hyp_params_dict.get('general')['output_dir']) / f'inference/tensor_flow/{name}_{ts}'
    )
    os.makedirs(current_run_dir)

    hyp_params_dict.get('inference')['output_dir'] = current_run_dir

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
        configs_file=pathlib.Path(hyp_params_dict.get('general')['configs_dir']) / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    # - Configure the GPU to run on
    choose_gpu(gpu_id=hyp_params_dict.get("inference")["gpu_id"], logger=logger)

    # - Build the model and optionally load the weights
    model, weights_loaded = get_model(
        mode='inference',
        hyper_parameters=hyp_params_dict,
        output_dir=hyp_params_dict.get('inference')['output_dir'],
        logger=None
    )
    print(model.summary())

    # - Compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=get_optimizer(args=hyp_params_dict),
        run_eagerly=True,
        metrics=hyp_params_dict.get('training')['metrics']
    )

    if args.infer_all:
        ckpt_fls = hyp_params_dict.get('inference')['checkpoint_files']
        results_dict = dict()
        for ckpt_fl in ckpt_fls:
            ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_fl)
            assert ckpt_loaded, f'Could not load weights from {ckpt_fl}!'

            ckpt_results = infer_all(
                model=model,
                hyper_parameters=hyp_params_dict,
                logger=logger
            )

            results_dict[ckpt_fl] = ckpt_results

        final_res_msgs = dict()
        for ckpt_fl, ckpt_res in results_dict.items():
            print_pretty_message(message=f'Checkpoint: {ckpt_fl}', delimiter_symbol='*')
            final_res_msg = print_results(results=ckpt_res, logger=logger)
            final_res_msgs[ckpt_fl] = final_res_msg

        for ckpt_fl, final_res_msg in final_res_msgs.items():
            print_pretty_message(message=f'Checkpoint: {ckpt_fl}', delimiter_symbol='*')

            if isinstance(logger, logging.Logger):
                info_log(logger=logger, message=final_res_msg)
            else:
                print(final_res_msg)
    else:
        ckpt_fl = hyp_params_dict.get('inference')['checkpoint_file']
        ckpt_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_fl)
        assert ckpt_loaded, f'Could not load weights from {ckpt_fl}!'
        results = infer_all(
            model=model,
            hyper_parameters=hyp_params_dict,
            logger=logger
        )
        final_res_msg = print_results(results=results, logger=logger)
