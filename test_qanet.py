import os
import pathlib
import time
import datetime


from configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger
from utils import test

if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'test/pytorch_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

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
    model_configs = get_model_configs(configs_file=MODEL_CONFIGS_FILE, logger=logger)
    trained_model = RibCage(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        input_image_shape=(args.image_height, args.image_width),
        conv2d_out_channels=model_configs.get('conv2d_blocks')['out_channels'],
        conv2d_kernel_sizes=model_configs.get('conv2d_blocks')['kernel_sizes'],
        fc_out_features=model_configs.get('fc_blocks')['out_features'],
        output_dir=args.output_dir,
        logger=logger
    ).to(device)

    chkpt_fl = pathlib.Path(args.tr_checkpoint_file)
    assert chkpt_fl.is_file(), f'Could not load model from \'{chkpt_fl}\' - file does not exist!'
    load_checkpoint(torch.load(chkpt_fl), trained_model)

    # -- GT
    # -*- Get the gold standard test data loader
    test_data_file = pathlib.Path(args.test_data_file)
    if test_data_file.is_file():
        test_model(
            model=trained_model,
            data_file=test_data_file,
            args=args,
            device=device,
            output_dir=output_dir,
            logger=logger
        )

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
