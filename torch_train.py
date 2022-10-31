import os
import pathlib
import time
import datetime


from global_configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from global_configs.general_configs import (
    MODEL_CONFIGS_FILE,
)
from utils.augs import train_augs, val_augs
from utils.aux_funcs import get_model_configs
from pytorch.custom.models import LitRibCage
from pytorch.utils.data_utils import get_data_loaders
from pytorch.utils.aux_funcs import get_optimizer
import warnings

__author__ = 'sidorov@post.bgu.ac.il'


warnings.filterwarnings("ignore")


if __name__ == '__main__':
    t_start = time.time()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current run
    current_run_dir = pathlib.Path(args.output_dir) / f'train/pytorch_{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print(f'''
    ======================================
    == Running train with PyTorch model ==
    ======================================
    ''')
    # - Build the model
    model_configs = get_model_configs(configs_file=MODEL_CONFIGS_FILE, logger=logger)
    model = LitRibCage(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        input_image_shape=(args.image_height, args.image_width),
        conv2d_out_channels=model_configs.get('conv2d_blocks')['out_channels'],
        conv2d_kernel_sizes=model_configs.get('conv2d_blocks')['kernel_sizes'],
        fc_out_features=model_configs.get('fc_blocks')['out_features'],
        optimizer=get_optimizer(
            algorithm=args.optimizer,
            args=dict(
                lr=args.optimizer_lr,
                lr_decay=args.optimizer_lr_decay,
                betas=(args.optimizer_beta_1, args.optimizer_beta_2),
                weight_decay=args.optimizer_weight_decay,
                momentum=args.optimizer_momentum,
                momentum_decay=args.optimizer_momentum_decay,
                dampening=args.optimizer_dampening,
                rho=args.optimizer_rho,
                nesterov=args.optimizer_nesterov,
                amsgrad=args.optimizer_amsgrad
            )
        ),
        output_dir=args.output_dir,
    )

    # - Load the data
    train_dl, val_dl = get_data_loaders(
        data_file=args.train_data_file,
        train_augs=train_augs,
        val_augs=val_augs,
        logger=logger
    )

    # - Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        auto_lr_find=True,
        gpus=2,
        auto_select_gpus=True,
        auto_scale_batch_size=True,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    # - Train model
    trainer.fit(model, train_dl, val_dl)

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
