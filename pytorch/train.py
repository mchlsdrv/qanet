
from .utils.torch_aux import get_device
from .utils.torch_utils import (
    train_model,
)
import warnings

__author__ = 'sidorov@post.bgu.ac.il'

warnings.filterwarnings("ignore")


def run(args, output_dir, logger):
    # - Configure the GPU to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)

    # - Train model
    trained_model = train_model(
        data_file=args.train_data_file,
        epochs=args.epochs,
        args=args,
        device=device,
        output_dir=output_dir,
        logger=logger
    )
