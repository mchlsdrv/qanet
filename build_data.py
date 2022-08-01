import argparse
import datetime
import os
import time

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import pathlib
import logging

from qanet import augs
from utils.aux_funcs import calc_jaccard, get_runtime, scan_files, plot_hist
from configs.global_configs import (
    TRAIN_DATA_DIR,
    GEN_DATA_DIR
)

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'

# - CONFIGS
DATA_TYPE = 'train'
SEG_DIR_POSTFIX = 'GT'
IMAGE_PREFIX = 't0'
SEG_PREFIX = 'man_seg0'
N_SAMPLES = 10000
MIN_J = 0.1
MAX_J = 0.9


def build_data_file(files: list, output_dir: pathlib.Path, n_samples=N_SAMPLES, plot_samples: bool = False):
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

            if MIN_J < jaccard < MAX_J:
                imgs.append(img)
                masks.append(mask)
                aug_masks.append(mask_aug)
                jaccards.append(jaccard)
                files_pbar.set_postfix(jaccard=f'{jaccard:.4f}')

    data = np.array(list(zip(imgs, masks, aug_masks, jaccards)), dtype=object)

    # - Save the data
    data_dir = output_dir / f'{DATA_TYPE}/{len(data)}_samples'

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
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(img, cmap='gray')
            ax[1].imshow(msk, cmap='gray')
            ax[2].imshow(msk_aug, cmap='gray')
            fig.suptitle(f'Jaccard: {j:.3f}')

            plt.savefig(samples_dir / f'{idx}.png')
            plt.close(fig)

            idx += 1

    # - Plot J histogram
    print(f'Plotting the histogram of the Js...')
    plot_hist(data=jaccards, hist_range=(0., 1., 0.1), bins=10, save_name=f'data dist ({len(data)} samples)', output_dir=data_dir, density=True)

    print(f'== Data generation took: {get_runtime(seconds=time.time() - t_start)} ==')

    return data


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data_dir', type=str, default=TRAIN_DATA_DIR, help='The path to the directory where the images and corresponding masks are stored')

    parser.add_argument('--gen_data_dir', type=str, default=GEN_DATA_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--n_samples', type=int, default=N_SAMPLES, help='The total number of samples to generate')

    parser.add_argument('--plot_samples', default=False, action='store_true', help=f'If to plot the samples images of the data')

    return parser


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Scan the files in the data dir
    fls = scan_files(
        root_dir=args.raw_data_dir,
        seg_dir_postfix=SEG_DIR_POSTFIX,
        image_prefix=IMAGE_PREFIX,
        seg_prefix=SEG_PREFIX
    )

    # - Build the data file
    build_data_file(
        files=fls,
        output_dir=args.gen_data_dir,
        n_samples=args.n_samples,
        plot_samples=args.plot_samples
    )
