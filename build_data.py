import time
import argparse
import os

import cv2
from tqdm import tqdm
import numpy as np
import pathlib
import logging

from utils import augs
from utils.aux_funcs import (
    calc_seg_measure,
    scan_files,
    plot_hist,
    get_file_name,
    plot_seg_error,
    get_range, load_image, float_2_str, str_2_float, get_runtime
)
from global_configs.general_configs import (
    N_SAMPLES,
    MIN_J,
    MAX_J,
    SEG_DIR_POSTFIX,
    SEG_PREFIX,
    IMAGE_PREFIX,
    TRAIN_DATA_DIR,
    SEG_SUB_DIR
)
GEN_DATA_DIR = pathlib.Path('../data/generated')
logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


def build_data(files: list, bins: np.ndarray, n_samples_in_range: int, output_dir: pathlib.Path, create_log: bool = False):
    # - Create ranges from bins
    assert len(bins) >= 2, f'bins must include at least 2 values, but were provided only {len(bins)} value/s!'
    bins_min, bins_max, bins_diff = np.min(bins), np.max(bins), bins[1] - bins[0]
    ranges = np.array(list(zip(np.arange(bins_min, bins_max + bins_diff, bins_diff), np.arange(bins_min + bins_diff, bins_max + 2 * bins_diff, bins_diff))))

    n_rngs = len(ranges)
    total_image_samples = n_rngs * n_samples_in_range

    mask_augs = augs.mask_augs()

    files_pbar = tqdm(files)
    for img_fl, seg_fl in files_pbar:
        t_start = time.time()
        # - Load the gt mask
        gt_msk = load_image(image_file=str(seg_fl), add_channels=True)

        # - Create a dedicated dir for the current image augs
        img_dir = output_dir / f'images/{get_file_name(path=str(img_fl))}'
        os.makedirs(img_dir, exist_ok=True)

        # - Create a counter array for the number of images in each range
        rng_cnt_arr = np.zeros(n_rngs, dtype=np.int16)

        # - Create a seg measure history array
        seg_msrs = np.array([])

        # - While not all the ranges for the current image have n_samples_in_range images
        rngs_done = sum(rng_cnt_arr == n_samples_in_range)
        while rngs_done < n_rngs:
            # - Augment the mask
            aug_res = mask_augs(image=gt_msk, mask=gt_msk)
            msk_aug = aug_res.get('mask')

            # - Calculate the seg measure
            seg_msr = calc_seg_measure(gt_msk, msk_aug)[0]

            # - Get the index and the range where the seg measure falls in
            rng_min, rng_max, rng_idx = get_range(value=seg_msr, ranges=ranges)

            # - If the seg measure is in the desired range - save it
            if MIN_J < seg_msr < MAX_J and rng_cnt_arr[rng_idx] < n_samples_in_range:
                # - Create a file name
                f_name = float_2_str(seg_msr) + '.tif'

                # - Save the mask
                cv2.imwrite(str(img_dir / f_name), msk_aug)

                # - Increase the images in range counter
                rng_cnt_arr[rng_idx] += 1

                # - Append the current seg measure to the seg measure history
                seg_msrs = np.append(seg_msrs, seg_msr)

            # - Update the number of ranges where there are n_samples_in_range samples
            rngs_done = sum(rng_cnt_arr == n_samples_in_range)

            # - Update the current samples number
            current_image_samples = sum(rng_cnt_arr)

            # - Update the progress bar
            files_pbar.set_postfix(done=f'{current_image_samples}/{total_image_samples} ({100 * current_image_samples / total_image_samples:.2f}%)', ranges=f'{rngs_done}/{n_rngs} ({100 * rngs_done / n_rngs:.2f}%)', current_run=f'range: [{rng_min:.2f}:{rng_max:.2f}), seg measure: {seg_msr:.4f}')

        if create_log:
            # - Load the image
            img = load_image(image_file=str(img_fl), add_channels=True)

            # - Create a dedicated dir for the current image augs
            img_log_dir = output_dir / f'log/{get_file_name(path=str(img_fl))}'
            os.makedirs(img_log_dir, exist_ok=True)

            # - Plot the histogram of the seg measures
            plot_hist(
                data=seg_msrs,
                bins=bins,
                save_file=img_log_dir / 'samples_distribution.png'
            )

            # - Plot the samples
            aug_msk_fls = [img_dir / str(fl) for fl in os.listdir(img_dir)]

            for aug_msk_fl in aug_msk_fls:
                # - Get the name of the mask, which is also the seg measure with its ground truth label
                seg_msr_str = get_file_name(path=str(aug_msk_fl))

                # - Load the augmented mask
                aug_msk = load_image(image_file=str(aug_msk_fl), add_channels=True)

                rng_min, rng_max, _ = get_range(value=str_2_float(seg_msr_str), ranges=ranges)

                # - Create a dedicated dir for the current seg measure range
                rng_img_log_dir = img_log_dir / f'[{float_2_str(rng_min)}-{float_2_str(rng_max)})'
                os.makedirs(rng_img_log_dir, exist_ok=True)

                # - Plot the image the GT mask and the augmented mask
                fig, ax = plot_seg_error(
                    image=img,
                    gt_mask=gt_msk,
                    pred_mask=aug_msk,
                    suptitle='GT (red) vs Pred (blue)',
                    title=f'Seg Measure = {str_2_float(seg_msr_str)}',
                    figsize=(20, 20),
                    save_file=rng_img_log_dir / f'{seg_msr_str}.png'
                )
        print(f'\n> Generation of {n_samples_in_range} augmented masks for image \'{img_fl}\' took {get_runtime(seconds=time.time() - t_start)}')


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', type=str, choices=['train', 'test'], help='The type of the data (i.e, train or test)')

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
        root_dir=TRAIN_DATA_DIR,
        seg_dir_postfix=SEG_DIR_POSTFIX,
        image_prefix=IMAGE_PREFIX,
        seg_prefix=SEG_PREFIX,
        seg_sub_dir=SEG_SUB_DIR
    )

    # - Build the data file
    if fls:
        build_data(
            files=fls,
            bins=np.arange(0.0, 1.0, 0.1),
            n_samples_in_range=10,
            output_dir=GEN_DATA_DIR,
            create_log=True
        )
    else:
        print(f'No files to generate data from !')
