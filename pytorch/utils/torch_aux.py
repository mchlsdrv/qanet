import logging.config

import logging

import torch


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_device(gpu_id: int = 0, logger: logging.Logger = None):
    n_gpus = torch.cuda.device_count()

    print('Available GPUs:')
    print(f'\t- Number of GPUs: {n_gpus}')
    device = 'cpu'
    if n_gpus > 0:
        try:
            if -1 < gpu_id < n_gpus - 1:
                print(f'Setting GPU to: {gpu_id}')

                device = f'cuda:{gpu_id}'

                print(f'''
    ========================
    == Running on {device}  ==
    ========================
                ''')
            elif gpu_id > n_gpus - 1:

                device = f'cuda'
                print(f'''
    =========================
    == Running on all GPUs ==
    =========================
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
