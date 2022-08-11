import os
import pathlib
import logging.config

from configs.general_configs import SEG_PREFIX, IMAGE_PREFIX, SEG_DIR_POSTFIX
from .. utils.augs import inference_augs
from utils.aux_funcs import err_log, scan_files

from . utils.tf_utils import (
    choose_gpu,
    get_model,
)

from . utils.tf_data_utils import (
    get_data_loaders,
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


def run(args, output_dir, logger: logging.Logger = None):
    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # MODEL
    # -1- Build the model and optionally load the weights
    trained_model = None
    try:
        trained_model, weights_loaded = get_model(
            model_configs=dict(
                input_image_dims=(args.image_height, args.image_width),
                kernel_regularizer=dict(
                    type=args.kernel_regularizer_type,
                    l1=args.kernel_regularizer_l1,
                    l2=args.kernel_regularizer_l2,
                    factor=args.kernel_regularizer_factor,
                    mode=args.kernel_regularizer_mode
                ),
                activation=dict(
                    type=args.activation,
                    max_value=args.activation_relu_max_value,
                    negative_slope=args.activation_relu_negative_slope,
                    threshold=args.activation_relu_threshold,
                    alpha=args.activation_leaky_relu_alpha
                )
            ),
            compilation_configs=dict(
                algorithm=args.optimizer,
                learning_rate=args.optimizer_lr,
                rho=args.optimizer_rho,
                beta_1=args.optimizer_beta_1,
                beta_2=args.optimizer_beta_2,
                amsgrad=args.optimizer_amsgrad,
                momentum=args.optimizer_momentum,
                nesterov=args.optimizer_nesterov,
                centered=args.optimizer_centered,
            ),
            checkpoint_dir=pathlib.Path(args.tf_checkpoint_dir),
            logger=logger
        )
    except Exception as err:
        err_log(logger=logger, message=f'{err}')

    if trained_model is not None and weights_loaded:
        # - Get the inference data loader
        data_tuples = scan_files(
            root_dir=args.inference_data_dir,
            seg_dir_postfix=SEG_DIR_POSTFIX,
            image_prefix=IMAGE_PREFIX,
            seg_prefix=SEG_PREFIX
        )

        # - Create the DataLoader object
        # - Get the train and the validation data loaders
        _, _, _, inf_dl = get_data_loaders(
            data=data_tuples,
            train_batch_size=0,
            val_prop=0,
            train_augs=None,
            val_augs=None,
            test_augs=None,
            inf_augs=inference_augs,
            logger=logger
        )

        # - Inference -
        preds = trained_model.infer(
            inf_dl
        )
        print(preds)
        print(f'''
        PREDICTIONS:
            mean: {preds.mean():.3f} +/- {preds.std():.4f}
        ''')
