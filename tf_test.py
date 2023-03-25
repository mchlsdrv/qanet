import os
import pathlib
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml

from tensor_flow.custom.tf_callbacks import write_figure_to_tensorboard
from tensor_flow.utils.tf_utils import (
    choose_gpu,
    test_model
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message,
    print_results, check_pathable, str_2_path,
)

from utils.visual_funcs import (
    get_scatter_plot_figure,
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_test(model, hyper_parameters: dict):
    t_start = time.time()

    data_name = ''
    if hyper_parameters.get('test')['test_sim']:
        hyper_parameters.get('test')['data_name'] = 'SIM'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_sim+']
    elif hyper_parameters.get('test')['test_gowt1']:
        hyper_parameters.get('test')['data_name'] = 'GOWT1'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_gowt1']
    elif hyper_parameters.get('test')['test_hela']:
        hyper_parameters.get('test')['data_name'] = 'HeLa'
        hyper_parameters.get('test')['dataframe_file'] = hyper_parameters.get('test')['dataframe_file_hela']

    # hyper_parameters.get('test')['data_name'] += hyper_parameters.get('test')['type']

    data_name = hyper_parameters.get('test')['data_name']
    if data_name:
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

        # assert hyper_parameters.get('test')['checkpoint_dir'] is not None, 'Have no checkpoint to test!'

        print_pretty_message(
            message=f'Running test with TensorFlow model from {hyper_parameters.get("test")["checkpoint_dir"]}',
            delimiter_symbol='='
        )

        # - Configure the GPU to run on
        choose_gpu(gpu_id=hyper_parameters.get('test')['gpu_id'], logger=logger)

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

        fig, rho, p, mse = get_scatter_plot_figure(
            x=x,
            y=y,
            plot_type='test',
            logger=logger)

        # - Create the train file writer
        file_writer = tf.summary.create_file_writer(
            str(current_run_dir / f"test - {hyper_parameters.get('test')['type']}"))

        write_figure_to_tensorboard(writer=file_writer, figure=fig, tag=f'{data_name}_test', step=0)

        fig.savefig(current_run_dir / f"{hyper_parameters.get('test')['data_name']}_scatter.png")
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
    else:
        print_pretty_message(message=f'No test data was chosen! Aborting...')


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyperparameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyperparameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)

    hyp_params_dict.get('test')['type'] = 'custom'

    run_test(model=None, hyper_parameters=hyp_params_dict)
