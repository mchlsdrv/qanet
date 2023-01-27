import os
import pathlib
import time
import datetime
import yaml
from tensor_flow.utils.tf_utils import (
    choose_gpu,
    test_model
)
from utils.aux_funcs import (
    get_arg_parser,
    get_runtime,
    get_logger,
    update_hyper_parameters, print_pretty_message, get_scatter_plot_figure,
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
    current_run_dir = pathlib.Path(hyp_params_dict.get('general')['output_dir']
                                   ) / f'test/tensor_flow_{args.name}_{ts}'
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
        configs_file=pathlib.Path(hyp_params_dict.get('general')['configs_dir']
                                  ) / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print_pretty_message(
        message=f'Running test with TensorFlow model from '
                f'{hyp_params_dict.get("test")["checkpoint_dir"]}',
        delimiter_symbol='='
    )

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    test_res_df = test_model(
        hyper_parameters=hyp_params_dict,
        output_dir=current_run_dir,
        logger=logger
    )

    # - Plot the scatter plot of the test vs gt
    fig, rho, p, mse = get_scatter_plot_figure(
        x=test_res_df.loc[:, 'seg_score'].values,
        y=test_res_df.loc[:, 'pred_seg_score'].values,
        plot_type='test', logger=logger)
    fig.savefig(current_run_dir / 'test_results_scatter_plot.png')

    # - Save the results
    test_res_df.to_csv(current_run_dir / 'test_results.csv')

    print_pretty_message(
        message=f'Total runtime: {get_runtime(seconds=time.time() - t_start)}',
        delimiter_symbol='='
    )
