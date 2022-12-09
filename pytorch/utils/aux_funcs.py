import logging
import logging.config
from functools import partial

import torch

from global_configs.general_configs import OPTIMIZER_EPS


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_device(gpu_id: int = 0, logger: logging.Logger = None):
    n_gpus = torch.cuda.device_count()

    print('> Available GPUs:')
    print(f'\t- Number of GPUs: {n_gpus}\n')
    device = 'cpu'
    if n_gpus > 0:
        try:
            if -1 < gpu_id < n_gpus - 1:
                print(f'> Setting GPU to: {gpu_id}\n')

                device = f'cuda:{gpu_id}'

                print(f'''
    ========================
    == Running on {device}  ==
    ========================
                ''')
            elif gpu_id < 0:
                device = 'cpu'
                print(f'''
    ====================
    == Running on CPU ==
    ====================
                        ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)

    return device


def get_optimizer(algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'sgd':
        optimizer = partial(
            torch.optim.SGD,
            lr=args.get('lr'),
            momentum=args.get('momentum'),
            weight_decay=args.get('weight_decay'),
            nesterov=args.get('nesterov'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adam':
        optimizer = partial(
            torch.optim.Adam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamw':
        optimizer = partial(
            torch.optim.AdamW,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'sparse_adam':
        optimizer = partial(
            torch.optim.SparseAdam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            torch.optim.NAdam,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            momentum_decay=args.get('momentum_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            torch.optim.Adamax,
            lr=args.get('lr'),
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adadelta':
        optimizer = partial(
            torch.optim.Adadelta,
            lr=args.get('lr'),
            rho=args.get('rho'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adagrad':
        optimizer = partial(
            torch.optim.Adadelta,
            lr=args.get('lr'),
            lr_decay=args.get('lr_decay'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    return optimizer
