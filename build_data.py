import argparse
import datetime
import os
import time

import cv2
from tqdm import tqdm
import numpy as np
import pathlib
import logging

from utils import augs
from utils.aux_funcs import calc_jaccard, get_runtime, scan_files, plot_hist, show_images
from configs.general_configs import (
    N_SAMPLES,
    MIN_J,
    MAX_J,
    INPUT_DATA_DIR,
    OUTPUT_DATA_DIR,
    SEG_DIR_POSTFIX,
    SEG_PREFIX,
    IMAGE_PREFIX,
    DATA_TYPE,
)

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


def build_data_file(files: list, output_dir: pathlib.Path, n_samples, min_j: int, max_j: int, plot_samples: bool = False):
    t_start = time.time()
    mask_augs = augs.mask_augs()
    imgs = []
    masks = []
    aug_masks = []
    jaccards = []

    n_passes = n_samples // len(files) if len(files) > 0 else 0
    for pass_idx in range(n_passes):
        print(f'\n== Pass: {pass_idx+1}/{n_passes} ==')
        files_pbar = tqdm(files)
        for img_fl, seg_fl in files_pbar:
            img = cv2.imread(str(img_fl), -1)
            mask = cv2.imread(str(seg_fl), -1)

            aug_res = mask_augs(image=img, mask=mask)
            img_aug, mask_aug = aug_res.get('image'), aug_res.get('mask')

            jaccard = calc_jaccard(mask, mask_aug)

            if min_j < jaccard < max_j:
                imgs.append(img)
                masks.append(mask)
                aug_masks.append(mask_aug)
                jaccards.append(jaccard)
                files_pbar.set_postfix(jaccard=f'{jaccard:.4f}')

    data = np.array(list(zip(imgs, masks, aug_masks, jaccards)), dtype=object)

    if len(data):
        # - Save the data
        data_dir = output_dir / f'{len(data)}_samples'

        # To avoid data overwrite
        if data_dir.is_dir():
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f'\'{data_dir}\' already exists! Data will be placed in \'{data_dir}/{ts}\'')
            data_dir = data_dir / ts

        os.makedirs(data_dir, exist_ok=True)
        np.save(str(data_dir / f'data.npy'), data, allow_pickle=True)
        print(f'Data was saved to \'{data_dir}/data.npy\'')

        if plot_samples:
            print(f'Plotting samples..')

            # - Plot samples
            samples_dir = data_dir / 'samples'
            os.makedirs(samples_dir, exist_ok=True)

            plot_pbar = tqdm(data)
            idx = 0
            for img, msk, msk_aug, j in plot_pbar:
                show_images(images=[img, msk, msk_aug], labels=['Image', 'Mask', 'Augmented Mask'], suptitle=f'Jaccard: {j:.4f}', figsize=(25, 10), save_file=samples_dir / f'{idx}.png')
                idx += 1

        # - Plot J histogram
        print(f'Plotting the histogram of the Js...')
        plot_hist(data=jaccards, hist_range=(0., 1., 0.1), bins=10, save_name=f'data dist ({len(data)} samples)', output_dir=data_dir, density=True)

        print(f'== Data generation took: {get_runtime(seconds=time.time() - t_start)} ==')
    else:
        print(f'No data was generated - no files were provided!')

    return data


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_dir', type=str, default=INPUT_DATA_DIR, help='The path to the directory where the train images and corresponding masks are stored')

    parser.add_argument('--output_data_dir', type=str, default=OUTPUT_DATA_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--seg_dir_postfix', type=str, default=SEG_DIR_POSTFIX, help='The postfix of the directory which holds the segmentations')

    parser.add_argument('--image_prefix', type=str, default=IMAGE_PREFIX, help='The prefix of the images')

    parser.add_argument('--seg_prefix', type=str, default=SEG_PREFIX, help='The prefix of the segmentations')

    parser.add_argument('--n_samples', type=int, default=N_SAMPLES, help='The total number of samples to generate')

    parser.add_argument('--min_j', type=float, default=MIN_J, help='The minimal allowed jaccard ')

    parser.add_argument('--max_j', type=float, default=MAX_J, help='The maximal allowed jaccard ')

    parser.add_argument('--plot_samples', default=False, action='store_true', help=f'If to plot the samples images of the data')

    return parser


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Scan the files in the data dir

    fls = scan_files(
        root_dir=args.input_data_dir,
        seg_dir_postfix=args.seg_dir_postfix,
        image_prefix=args.image_prefix,
        seg_prefix=args.seg_prefix
    )

    # - Build the data file
    if fls:
        build_data_file(
            files=fls,
            output_dir=args.output_data_dir,
            n_samples=args.n_samples,
            min_j=args.min_j,
            max_j=args.max_j,
            plot_samples=args.plot_samples,
        )
    else:
        print(f'No files to generate data from !')
