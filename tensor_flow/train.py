import os

from utils.aux_funcs import err_log
from .utils.tf_utils import (
    choose_gpu,
    train_model,
)

import warnings
warnings.filterwarnings("ignore")

'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


def run(args, output_dir, logger):
    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    trained_model = None
    try:
        trained_model = train_model(
            args=args,
            output_dir=output_dir,
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')
