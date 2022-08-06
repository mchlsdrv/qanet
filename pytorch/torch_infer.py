import pathlib

import torch

from configs.general_configs import MODEL_CONFIGS_FILE
from utils.aux_funcs import get_model_configs
from .custom.torch_models import RibCage
from .utils.torch_aux import get_device, load_checkpoint
from .utils.torch_utils import (
    infer
)
import warnings

__author__ = 'sidorov@post.bgu.ac.il'

warnings.filterwarnings("ignore")


def run(args, output_dir, logger):
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

    chkpt_fl = pathlib.Path(args.checkpoint_file)
    assert chkpt_fl.is_file(), f'Could not load model from {chkpt_fl}!'
    load_checkpoint(torch.load(chkpt_fl), model)

    # - INFERENCE -
    inference_data_dir = pathlib.Path(args.inference_data_dir)
    if inference_data_dir.is_file():
        infer(
            data_dir=inference_data_dir,
            model=model,
            device=device,
            output_dir=output_dir
        )
