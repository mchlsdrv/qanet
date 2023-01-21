import argparse
import os
import time

import cv2
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import numpy as np
import pathlib
import logging

import albumentations as A
import albumentations.augmentations.transforms as tr
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)

from utils import augs
from utils.aux_funcs import plot_hist, calc_seg_score, monitor_seg_error, get_ts, calc_histogram

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'

PLOT_SAMPLES = True
N_SAMPLES_TO_PLOT = 10
EROSION_SIZES = [1, 3, 5, 7, 9, 10, 11, 13, 15]
DILATION_SIZES = [1, 3, 5, 7, 9, 10, 11, 13, 15]
IMAGE_WIDTH = 419
IMAGE_HEIGHT = 419
ALPHA = IMAGE_WIDTH * 2
SIGMA = IMAGE_WIDTH * 0.15
ALPHA_AFFINE = IMAGE_WIDTH * 0.08
P_EROSION = 0.5
P_DILATION = 0.5
P_OPENING = 0.5
P_CLOSING = 0.5
P_ONE_OF = 0.5
P_ELASTIC = 0.5

SEG_DIR_POSTFIX = 'GT'
IMAGE_PREFIX = 't'
SEG_PREFIX = 'man_seg'
SEG_SUB_DIR = 'SEG'

CLAHE_CLIP_LIMIT = 0.8
CLAHE_TILE_GRID_SIZE = 8

OUTPUT_DIR = pathlib.Path('/home/sidorov/Projects/QANetV2/data/data_balance') / get_ts()
os.makedirs(OUTPUT_DIR)


def random_erosion(mask, **kwargs):
    # Shrinks the labels
    krnl = np.random.choice(EROSION_SIZES)
    mask = grey_erosion(mask, size=krnl)
    return mask


def random_dilation(mask, **kwargs):
    # "Fattens" the cell label
    krnl = np.random.choice(DILATION_SIZES)
    mask = grey_dilation(mask, size=krnl)
    return mask


def random_opening(mask, **kwargs):
    mask = random_dilation(random_erosion(mask))
    return mask


def random_closing(mask, **kwargs):
    # Disconnected labels are brought together
    mask = random_erosion(random_dilation(mask))
    return mask


def mask_augmentations():
    return A.Compose(
        [
            A.OneOf(
                [
                    tr.Lambda(
                        mask=random_erosion,
                        p=P_EROSION
                    ),
                    tr.Lambda(
                        mask=random_dilation,
                        p=P_DILATION
                    ),
                    tr.Lambda(
                        mask=random_opening,
                        p=P_OPENING
                    ),
                    tr.Lambda(
                        mask=random_closing,
                        p=P_CLOSING
                    ),
                ],
                p=P_ONE_OF
            ),
            A.ElasticTransform(
                alpha=ALPHA,
                sigma=SIGMA,
                alpha_affine=ALPHA_AFFINE,
                interpolation=cv2.INTER_LANCZOS4,
                approximate=True,
                same_dxdy=True,
                p=P_ELASTIC
            ),
        ]
    )


def transforms():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                p=1.
            ),
            A.ToFloat(p=1.),
        ]
    )


def calc_j_hists(files: list, trial_index: int, output_dir, normalize: bool = False, n_examples: int = 5, plot_examples: bool = False):
    t_start = time.time()
    image_mask_augs = augs.train_augs(crop_height=IMAGE_HEIGHT, crop_width=IMAGE_WIDTH)
    trnsfrms = transforms()
    mask_augs = mask_augmentations()

    imgs = []
    gt_msks = []
    print(f' - Loading images ... ')
    files_pbar = tqdm(files)
    for img_fl, gt_msk_fl in files_pbar:
        # - Load and add the image
        img = cv2.imread(str(img_fl), -1)
        img = img / img.max()
        img = np.expand_dims(img, axis=-1)
        imgs.append(img)

        # - Load and add the mask
        gt_msk = cv2.imread(str(gt_msk_fl), -1)
        gt_msk = np.expand_dims(gt_msk, axis=-1)
        gt_msks.append(gt_msk)

    print(f'''
    ============================
    == Calculating Statistics ==
    ============================
    ''')
    print(f'- Augmenting GT masks ...')
    j_hists = []
    trnsfrm_imgs = []
    trnsfrm_gt_msks = []
    aug_dfrmd_msks = []
    img_msk_data = list(zip(imgs, gt_msks))
    np.random.shuffle(img_msk_data)
    img_msk_pbar = tqdm(img_msk_data)
    for img, msk in img_msk_pbar:
        trnsfrm_res = trnsfrms(image=img, mask=msk)
        trnsfrm_img, trnsfrm_gt_msk = trnsfrm_res.get('image'), trnsfrm_res.get('mask')

        # - Deform the augmented gt mask
        aug_res = mask_augs(image=trnsfrm_img, mask=trnsfrm_gt_msk)
        aug_dfrmd_msk = aug_res.get('mask')

        trnsfrm_imgs.append(trnsfrm_img)
        trnsfrm_gt_msks.append(trnsfrm_gt_msk)
        aug_dfrmd_msks.append(aug_dfrmd_msk)

    # - Convert list to numpy array
    trnsfrm_gt_msks = np.array(trnsfrm_gt_msks)
    aug_dfrmd_msks = np.array(aug_dfrmd_msks)

    # - Calculate jaccards for the augmented samples
    js = calc_seg_score(gt_masks=trnsfrm_gt_msks, pred_masks=aug_dfrmd_msks)

    # - Calculate the histogram
    print(f'- Plotting J histogram ...')
    ts = get_ts()
    bins = np.arange(0.0, 1.1, 0.1)
    j_hist, _ = calc_histogram(
        data=js,
        bins=bins,
        normalize=normalize
    )
    j_hists.append(j_hist)

    # - Plot samples
    if PLOT_SAMPLES:
        print(f'- Plotting seg measure histogram..')
        plot_hist(
            data=js,
            bins=bins,
            save_file=OUTPUT_DIR / f'trial_{trial_index}/j_distribution.png' if isinstance(OUTPUT_DIR, pathlib.Path) else None
        )
        print(f'- Plotting samples..')
        monitor_seg_error(
            gt_masks=trnsfrm_gt_msks,
            pred_masks=aug_dfrmd_msks,
            seg_measures=js,
            images=np.array(trnsfrm_imgs),
            n_samples=N_SAMPLES_TO_PLOT,
            figsize=(20, 10),
            save_dir=OUTPUT_DIR / f'trial_{trial_index}/examples'
        )

    params = pd.DataFrame(
        dict(
            erosion_sizes=[EROSION_SIZES],
            dilation_sizes=[DILATION_SIZES],
            alpha_fctr=ALPHA / IMAGE_WIDTH,
            sigma_fctr=SIGMA / IMAGE_WIDTH,
            alpha_affine_fctr=ALPHA_AFFINE / IMAGE_WIDTH,
            p_erosion=P_EROSION,
            p_dilation=P_DILATION,
            p_opening=P_OPENING,
            p_closing=P_CLOSING,
            p_elastic=P_ELASTIC,
            p_one_of=P_ONE_OF
        ), index=[0]
    )
    params.to_csv(OUTPUT_DIR / f'trial_{trial_index}/params.csv')

    return np.array(j_hists).flatten()


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', type=str, choices=['train', 'test'], help='The type of the data (i.e, train or test)')

    parser.add_argument('--seg_dir_postfix', type=str, default=SEG_DIR_POSTFIX, help='The postfix of the directory which holds the segmentations')

    parser.add_argument('--image_prefix', type=str, default=IMAGE_PREFIX, help='The prefix of the images')

    parser.add_argument('--seg_prefix', type=str, default=SEG_PREFIX, help='The prefix of the segmentations')

    parser.add_argument('--n_samples', type=int, default=100, help='The total number of samples to generate')

    parser.add_argument('--plot_samples', default=False, action='store_true', help=f'If to plot the samples images of the data')

    return parser


def objective(trial, files):
    global EROSION_SIZES
    global DILATION_SIZES
    global ALPHA
    global ALPHA_AFFINE
    global SIGMA
    global P_EROSION
    global P_DILATION
    global P_OPENING
    global P_CLOSING
    global P_ONE_OF

    EROSION_SIZES = []
    n_erosion_sizes = trial.suggest_int('n_erosion_sizes', 3, 10)
    for n in range(n_erosion_sizes):
        EROSION_SIZES.append(2 * n + 1)

    DILATION_SIZES = []
    n_dilation_sizes = trial.suggest_int('n_dilation_sizes', 3, 10)
    for n in range(n_dilation_sizes):
        DILATION_SIZES.append(2 * n + 1)

    alpha_fctr = trial.suggest_float('alpha_factor', 1.0, 5.0)
    ALPHA = IMAGE_WIDTH * alpha_fctr

    alpha_affine_fctr = trial.suggest_float('alpha_affine_factor', 0.01, 0.1)
    ALPHA_AFFINE = IMAGE_WIDTH * alpha_affine_fctr

    sigma_fctr = trial.suggest_float('sigma_factor', 0.1, 1.0)
    SIGMA = IMAGE_WIDTH * sigma_fctr

    P_EROSION = trial.suggest_float('p_erosion', 0.0, 1.0)
    P_DILATION = trial.suggest_float('p_dilation', 0.0, 1.0)
    P_OPENING = trial.suggest_float('p_opening', 0.0, 1.0)
    P_CLOSING = trial.suggest_float('p_closing', 0.0, 1.0)
    P_ELASTIC = trial.suggest_float('p_elastic', 0.0, 1.0)
    P_ONE_OF = trial.suggest_float('p_one_of', 0.0, 1.0)

    j_dist = calc_j_hists(files=files, trial_index=trial.number, output_dir=OUTPUT_DIR, normalize=True)

    mse = np.mean(np.square(np.ones_like(j_dist) * 0.5 - j_dist))
    print(f'''
    Stats for trial #{trial.number}:
        > Params:
            - EROSION SIZES: {EROSION_SIZES}
            - DILATION SIZES: {DILATION_SIZES}
            - ALPHA (factor = {alpha_fctr}): {ALPHA}
            - ALPHA AFFINE (factor = {alpha_affine_fctr}): {ALPHA_AFFINE}
            - SIGMA (factor = {sigma_fctr}): {SIGMA}
            - P EROSION: {P_EROSION}
            - P DILATION: {P_DILATION}
            - P OPENING: {P_OPENING}
            - P CLOSING: {P_CLOSING}
            - P ELASTIC: {P_ELASTIC}
            - P ONE OF: {P_ONE_OF}
        > Results:
            - MSE: {mse} 
    ''')
    if trial.study.best_trials:
        print(f'''
        ===================================
        > Best Trial: #{trial.study.best_trial.number}, mse: {trial.study.best_trial.value:.4f}
        ===================================
        ''')
    return mse


if __name__ == '__main__':
    gt_train = '/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-11-25_21-56-39/train/plots/scatter/metadata/gt_seg_measures_epoch_527.npy'
    pred_train = '/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-11-25_21-56-39/train/plots/scatter/metadata/pred_seg_measures_epoch_527.npy'
    gt_seg_measure_train = np.load(gt_train)
    pred_seg_measure_train = np.load(pred_train)
    r_train, _ = pearsonr(gt_seg_measure_train, pred_seg_measure_train)

    gt_val = '/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-11-25_21-56-39/validation/plots/scatter/metadata/gt_seg_measures_epoch_527.npy'
    pred_val = '/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-11-25_21-56-39/validation/plots/scatter/metadata/pred_seg_measures_epoch_527.npy'
    gt_seg_measure_val = np.load(gt_val)
    pred_seg_measure_val = np.load(pred_val)
    r_val, _ = pearsonr(gt_seg_measure_val, pred_seg_measure_val)

    mse_train = np.mean(np.square(gt_seg_measure_train - pred_seg_measure_train))
    mse_val = np.mean(np.square(gt_seg_measure_val - pred_seg_measure_val))

    print(f'Train R - {r_train:.4f}\nTrain MSE - {mse_train:.4f}')
    print(f'Val R - {r_val:.4f}\nVal MSE - {mse_val:.4f}')
    # - Get the argument parser
    # parser = get_arg_parser()
    # args = parser.parse_args()
    #
    # # - Scan the files in the data dir
    #
    # fl_tupls = scan_files(
    #     root_dir=TRAIN_DATA_DIR,
    #     seg_dir_postfix=SEG_DIR_POSTFIX,
    #     image_prefix=IMAGE_PREFIX,
    #     seg_prefix=SEG_PREFIX,
    #     seg_sub_dir=SEG_SUB_DIR
    # )
    # np.random.shuffle(fl_tupls)
    # study = opt.create_study(direction='minimize')
    # study.optimize(partial(objective, files=fl_tupls[:100]), n_trials=10000)
    #
    # best_trial = study.best_trial
    # best_trial_params = best_trial.params
    # print(f'''
    # ========================================================================================================
    # ============================================ SUMMARY ===================================================
    # ========================================================================================================
    #     Best trial:
    #         - MSE: {best_trial.values[0]}
    #         - Params:
    #             > Number of erosion sizes: {best_trial_params.get('n_erosion_sizes')}
    #             > DILATION SIZES: {best_trial_params.get('n_dilation_sizes')}
    #             > ALPHA (factor = {best_trial_params.get('alpha_factor')}): {IMAGE_WIDTH * best_trial_params.get('alpha_factor')}
    #             > ALPHA AFFINE (factor = {best_trial_params.get('alpha_affine_factor')}): {IMAGE_WIDTH * best_trial_params.get('alpha_affine_factor')}
    #             > SIGMA (factor = {best_trial_params.get('sigma_factor')}): {IMAGE_WIDTH * best_trial_params.get('sigma_factor')}
    #             > P EROSION: {best_trial_params.get('p_erosion')}
    #             > P DILATION: {best_trial_params.get('p_dilation')}
    #             > P OPENING: {best_trial_params.get('p_opening')}
    #             > P CLOSING: {best_trial_params.get('p_closing')}
    #             > P ELASTIC: {best_trial_params.get('p_elastic')}
    #             > P ONE OF: {best_trial_params.get('p_one_of')}
    # ========================================================================================================
    # ========================================================================================================
    # ''')
