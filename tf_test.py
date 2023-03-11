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
    print_results,
)

from utils.visual_funcs import (
    get_scatter_plot_figure,
)
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

    data_name = ''
    if args.test_sim:
        data_name = 'SIM'
        hyp_params_dict.get('test')['dataframe_file'] = hyp_params_dict.get('test')['dataframe_file_sim+']
    if args.test_gowt1:
        data_name = 'GOWT1'
        hyp_params_dict.get('test')['dataframe_file'] = hyp_params_dict.get('test')['dataframe_file_gowt1']
    elif args.test_hela:
        data_name = 'HeLa'
        hyp_params_dict.get('test')['dataframe_file'] = hyp_params_dict.get('test')['dataframe_file_hela']

    if data_name:
        current_run_dir = args.output_dir
        if current_run_dir is None:
            # - Create the directory for the current run
            exp_name = args.experiment_name
            current_run_dir = pathlib.Path(hyp_params_dict.get(
                'general')['output_dir']) / f'test/tensor_flow/{data_name}_{exp_name}_{ts}'
            os.makedirs(current_run_dir)

        print_pretty_message(
            message=f'Current run dir was set to: {current_run_dir}',
            delimiter_symbol='='
        )

        # - Save the updated hyperparameters to the current run directory
        yaml.dump(
            hyp_params_dict,
            (current_run_dir / 'test_hyper_params.yml').open(mode='w')
        )

        # - Configure the logger
        logger = get_logger(
            configs_file=pathlib.Path(hyp_params_dict.get('general')['configs_dir']) / 'logger_configs.yml',
            save_file=current_run_dir / f'logs.log'
        )

        if args.checkpoint_dir is not None:
            hyp_params_dict.get("test")["checkpoint_dir"] = args.checkpoint_dir

        print_pretty_message(
            message=f'Running test with TensorFlow model from {hyp_params_dict.get("test")["checkpoint_dir"]}',
            delimiter_symbol='='
        )

        # - Configure the GPU to run on
        choose_gpu(gpu_id=args.gpu_id, logger=logger)

        # - Run the test
        test_res_df = test_model(
            hyper_parameters=hyp_params_dict,
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
        file_writer = tf.summary.create_file_writer(str(current_run_dir / 'test'))

        write_figure_to_tensorboard(writer=file_writer, figure=fig, tag=f'{data_name}_test', step=0)

        fig.savefig(current_run_dir / 'test_scatter.png')
        plt.close(fig)

        # fig, ax = get_simple_scatter_plot_figure(
        #     x=x,
        #     y=y,
        #     xlabel='GT Quality Value',
        #     ylabel='Estimated Quality Value',
        #     save_file=current_run_dir / 'gt vs estimated scatter plot.png'
        # )

        # - Save the results
        test_res_df.to_csv(current_run_dir / f'{data_name}_test_results.csv')

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
