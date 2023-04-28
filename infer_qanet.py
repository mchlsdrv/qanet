import os
import pathlib
import time
import datetime

from configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger
from utils import infer

if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'inference/pytorch_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    ==========================================
    == Running inference with PyTorch model ==
    ==========================================
    ''')
    # - Configure the GPU to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)

    # - Build the model
    model_configs = get_model_configs(configs_file=MODEL_CONFIGS_FILE, logger=logger)
    model = RibCage(
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
    load_checkpoint(torch.load(chkpt_fl), model)

    # - INFERENCE -
    inf_data_dir = pathlib.Path(args.inference_data_dir)
    assert inf_data_dir.is_dir(), f'The \'{inf_data_dir}\' directory does not exist!'
    # - Get file tuples
    data_fls = scan_files(root_dir=inf_data_dir, seg_dir_postfix=SEG_DIR_POSTFIX, image_prefix=IMAGE_PREFIX, seg_prefix=SEG_PREFIX)

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
