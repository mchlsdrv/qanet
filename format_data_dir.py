import argparse
import os
import pathlib

DATA_ROOT_DIR = '/home/sidorov/Projects/QANetV2/data/UNSW-AU-Fluo-N2DH-GOWT1'

SEG_DIR_POSTFIX = 'RES'
IMAGE_PREFIX = 't0'
SEG_PREFIX = 'mask0'

CURRENT_SEG_DIR_POSTFIX = 'RES'
CURRENT_IMAGE_PREFIX = 't0'
CURRENT_SEG_PREFIX = 'mask0'


def rename_files(data_root_dir: pathlib.Path or str, current_image_prefix: str,
                 current_seg_prefix: str):
    for root, dirs, files in os.walk(data_root_dir):
        for file in files:
            if file.find(current_image_prefix) > -1:
                os.rename(f'{root}/{file}',
                          f'{root}/'
                          f'{file.replace(current_image_prefix, IMAGE_PREFIX)}')
            elif file.find(current_seg_prefix) > -1:
                os.rename(f'{root}/{file}',
                          f'{root}/'
                          f'{file.replace(current_seg_prefix, SEG_PREFIX)}')


def rename_dirs(data_root_dir: pathlib.Path or str,
                current_seg_dir_postfix: str):
    for root, dirs, files in os.walk(data_root_dir):
        for dir in dirs:
            if dir.find(current_seg_dir_postfix) > -1:
                os.rename(
                    f'{root}/{dir}',
                    f'{root}/'
                    f'{dir.replace(current_seg_dir_postfix, SEG_DIR_POSTFIX)}')


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir', type=str, default=DATA_ROOT_DIR,
                        help='The path to the root directory')

    parser.add_argument('--current_seg_dir_postfix', type=str,
                        default=CURRENT_SEG_DIR_POSTFIX,
                        help='The current postfix of the directory which holds'
                             ' the segmentations')

    parser.add_argument('--current_image_prefix', type=str,
                        default=CURRENT_IMAGE_PREFIX,
                        help='The current prefix of the images')

    parser.add_argument('--current_seg_prefix', type=str,
                        default=CURRENT_SEG_PREFIX,
                        help='The current prefix of the segmentations')

    return parser


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    rename_files(
        data_root_dir=args.data_root_dir,
        current_image_prefix=args.current_image_prefix,
        current_seg_prefix=args.current_seg_prefix)

    rename_dirs(
        data_root_dir=args.data_root_dir,
        current_seg_dir_postfix=args.current_seg_dir_postfix)
