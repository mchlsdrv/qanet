import pathlib

from . utils.torch_utils import (
    train_model,
    test_model,
    get_device
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
        save_dir=output_dir,
        logger=logger
    )

    # - TEST -
    # -- GT
    # -*- Get the gold standard test data loader
    gt_test_data_file = pathlib.Path(args.gt_test_data_file)
    if gt_test_data_file.is_file():
        test_model(
            model=trained_model,
            data_file=gt_test_data_file,
            args=args,
            device=device,
            save_dir=output_dir,
            logger=logger
        )
    # -- ST
    # -*- Get the silver standard test data loader
    # st_test_data_file = pathlib.Path(args.st_test_data_file)
    # if st_test_data_file.is_file():
    #     test_model(
    #         model=trained_model,
    #         data_file=st_test_data_file,
    #         args=args,
    #         device=device,
    #         save_dir=output_dir,
    #         logger=logger
    #     )
