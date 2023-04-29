import os
import pathlib
import time
import datetime

import yaml
import matplotlib as mpl

from configs.general_configs import CONFIGS_DIR
from utils.aux_funcs import get_arg_parser, get_runtime, get_logger, update_hyper_parameters, get_model, \
    print_pretty_message

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from utils.augs import (
    train_augs, val_augs
)
import warnings
from utils.data_utils import (
    get_data_loaders
)

__author__ = 'sidorov@post.bgu.ac.il'


warnings.filterwarnings("ignore")


if __name__ == '__main__':
    t_start = time.time()

    # - Get the current timestamp to be used for the run differentiate the run
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Get hyper-parameters
    hyp_params_fl = pathlib.Path(args.hyper_params_file)
    hyp_params_dict = yaml.safe_load(hyp_params_fl.open(mode='r').read())

    # - Update the hyper-parameters with the parsed arguments
    update_hyper_parameters(hyper_parameters=hyp_params_dict, arguments=args)
    if not hyp_params_dict.get('general')['debug']:
        mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by the wrong thread" exception

    # - Create the directory for the current run
    if hyp_params_dict.get('training')['load_checkpoint']:
        current_run_dir = pathlib.Path(hyp_params_dict.get('training')['checkpoint_file']).parent
        dir_name = current_run_dir.name
    else:
        crp_w = hyp_params_dict.get('augmentations')['crop_width']
        crp_h = hyp_params_dict.get('augmentations')['crop_height']
        dir_name = f'{hyp_params_dict.get("general")["name"]}_{ts}'
        current_run_dir = pathlib.Path(
            hyp_params_dict.get('general')['output_dir']) / f'pytorch/train/{crp_h}x{crp_w}/{dir_name}'
        os.makedirs(current_run_dir)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    print_pretty_message(message=f'Running train with PyTorch model')

    # - Build the model
    model = get_model(hyper_parameters=hyp_params_dict)

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
        plugins=DDPStrategy(find_unused_parameters=False),
    )

    # - Train model
    trainer.fit(model, train_dl, val_dl)

    print(f'\n== Total runtime: {get_runtime(seconds=time.time() - t_start)} ==\n')
