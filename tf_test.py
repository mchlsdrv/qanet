import logging
import os
import pathlib
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml

import warnings
from tensor_flow.utils.tf_utils import (
    choose_gpu,
    test_model, get_model, get_optimizer
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message,
    print_results, check_pathable, str_2_path, get_metrics,
)

from utils.visual_funcs import (
    get_simple_scatter_plot_figure,
)
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore")

plt.style.use('seaborn')  # <= using the seaborn plot style


def update_test_hyper_parameters(hyper_parameters: dict):
    if hyper_parameters.get('test')['test_sim']:
        hyper_parameters.get('test')['data_name'] = 'SIM'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_sim+']
    elif hyper_parameters.get('test')['test_gowt1']:
        hyper_parameters.get('test')['data_name'] = 'GOWT1'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_gowt1']
    elif hyper_parameters.get('test')['test_hela']:
        hyper_parameters.get('test')['data_name'] = 'HeLa'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_hela']


def run_test(model, hyper_parameters: dict):
    t_start = time.time()
    test_res_df = None
    data_name = hyper_parameters.get('test')['data_name']

    current_run_dir = hyper_parameters.get('test')['output_dir']
    if current_run_dir == '':
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # - Create the directory for the current run
        exp_name = hyper_parameters.get('test')['name']
        current_run_dir = pathlib.Path(
            pathlib.Path(hyper_parameters.get('general')['output_dir']) /
            f'test/tensor_flow/{data_name}_{exp_name}_{ts}')
        os.makedirs(current_run_dir)
    elif check_pathable(current_run_dir):
        current_run_dir = str_2_path(path=current_run_dir)

    print_pretty_message(
        message=f'Current run dir was set to: {current_run_dir}',
        delimiter_symbol='='
    )

    # - Save the updated hyperparameters to the current run directory
    yaml.dump(
        hyper_parameters,
        (current_run_dir / 'test_hyper_params.yml').open(mode='w')
    )

    # - Configure the logger
    logger = get_logger(
        configs_file=pathlib.Path(hyper_parameters.get('general')['configs_dir']) / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print_pretty_message(
        message=f'Running test with TensorFlow model from {hyper_parameters.get("test")["checkpoint_dir"]}',
        delimiter_symbol='='
    )

    # - Run the test
    test_res_df = test_model(
        model=model,
        hyper_parameters=hyper_parameters,
        output_dir=current_run_dir,
        logger=logger
    )

    # - Plot the scatter plot of the test vs gt
    x = test_res_df.loc[:, 'seg_score'].values
    y = test_res_df.loc[:, 'pred_seg_score'].values

    rho, p, mse = get_metrics(x=x, y=y)

    fig, ax = get_simple_scatter_plot_figure(
        x=x, y=y,
        xlabel='True', ylabel='Predicted',
        xticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), xlim=(0.0, 1.0),
        yticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ylim=(0.0, 1.0),
        figsize=(20, 20),
        save_file=current_run_dir / f"{hyper_parameters.get('test')['data_name']}_scatter.png"
    )

    plt.close(fig)

    # - Save the results
    test_res_df.to_csv(current_run_dir / f'{data_name}_test_results.csv', index=False)

    print_pretty_message(
        message=f'Test Results on {data_name} Data',
        delimiter_symbol='*'
    )

    # - Print the results
    print_results(results=test_res_df, rho=rho, p=p, mse=mse)

    print_pretty_message(
        message=f'Total runtime: {get_runtime(seconds=time.time() - t_start)}',
        delimiter_symbol='='
    )

    return test_res_df


def test_all(model, hyper_parameters: dict, logger: logging.Logger = None):
    # - Configure the GPU to run on
    choose_gpu(gpu_id=hyper_parameters.get('test')['gpu_id'], logger=logger)

    if model is None:
        # -1- Build the model and optionally load the weights
        model, weights_loaded = get_model(
            mode='test',
            hyper_parameters=hyper_parameters,
            output_dir=hyper_parameters.get('test')['output_dir'],
            logger=logger
        )

        # - Compile the model
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=get_optimizer(args=hyper_parameters),
            run_eagerly=True,
            metrics=hyper_parameters.get('training')['metrics']
        )

    # - SIM+
    print_pretty_message(message='Testing the SIM+ Data')
    hyper_parameters.get('test')['test_sim'] = True
    update_test_hyper_parameters(hyper_parameters=hyper_parameters)
    sim_test_res_dict = run_test(model=model, hyper_parameters=hyper_parameters)
    hyper_parameters.get('test')['test_sim'] = False

    # - GOWT1
    print_pretty_message(message='Testing the GOWT1 Data')
    hyper_parameters.get('test')['test_gowt1'] = True
    update_test_hyper_parameters(hyper_parameters=hyper_parameters)
    gowt1_test_res_dict = run_test(model=model, hyper_parameters=hyper_parameters)
    hyper_parameters.get('test')['test_gowt1'] = False

    # - HeLa
    print_pretty_message(message='Testing the HeLa Data')
    hyper_parameters.get('test')['test_hela'] = True
    update_test_hyper_parameters(hyper_parameters=hyper_parameters)
    hela_test_res_dict = run_test(model=model, hyper_parameters=hyper_parameters)
    hyper_parameters.get('test')['test_hela'] = False

    return sim_test_res_dict, gowt1_test_res_dict, hela_test_res_dict


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyperparameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # - Create the directory for the current run
    exp_name = hyp_params_dict.get('test')['name']
    current_run_dir = pathlib.Path(
        pathlib.Path(hyp_params_dict.get('general')['output_dir']) / f'test/tensor_flow/{exp_name}_{ts}')
    os.makedirs(current_run_dir)
    hyp_params_dict.get('test')['output_dir'] = str(current_run_dir)

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    hyp_params_dict.get('test')['type'] = 'custom'

    test_all(model=None, hyper_parameters=hyp_params_dict)
