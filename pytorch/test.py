import pathlib

import torch

from configs.general_configs import MODEL_CONFIGS_FILE
from utils.aux_funcs import get_model_configs
from .custom.torch_models import RibCage
from .utils.torch_aux import get_device, load_checkpoint
from . utils.torch_utils import (
    test_model,
)
import warnings

__author__ = 'sidorov@post.bgu.ac.il'

warnings.filterwarnings("ignore")


def run(args, output_dir, logger):
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
