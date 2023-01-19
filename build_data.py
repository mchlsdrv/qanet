import time
import argparse
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
# from scipy.ndimage import grey_erosion
from tqdm import tqdm
import numpy as np
import pathlib
import logging

from utils import augs
from utils.aux_funcs import (
    calc_seg_score,
    scan_files,
    plot_hist,
    get_file_name,
    get_range,
    load_image,
    float_2_str,
    str_2_float,
    get_runtime,
    get_parent_dir_name,
    get_ts,
    get_split_data,
    check_unique_file,
    assert_pathable, get_image_mask_figure, split_instance_mask, instance_2_categorical, merge_categorical_masks
)
import multiprocessing as mp
from multiprocessing import Pool

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'

DATA_DIR = pathlib.Path('/home/sidorov/Projects/QANetV2/data/train/CytoPack')
OUTPUT_DIR = pathlib.Path('/media/oldrrtammyfs/Users/sidorov/qanet/generated/masks')
# OUTPUT_DIR = pathlib.Path('/home/sidorov/Projects/QANetV2/data/generated/masks')
CROP_WIDTH = 419
N_SAMPLES_IN_RANGE = 10
FILE_MIN_SAMPLES_PCT = 96
FILE_MAX_TIME = 3 * 60  # in seconds
MAX_CPUS = 30
MIN_J = 0.01
MAX_J = 0.99

SEG_DIR_POSTFIX = 'GT'
IMAGE_PREFIX = 't'
SEG_PREFIX = 'man_seg'
SEG_SUB_DIR = 'SEG'


def repaint_instance_segmentation(mask: np.ndarray):
    msk = mask.astype(np.int8)

    # - Get the initial labels excluding the background
    lbls = np.unique(msk)
    lbls = lbls[lbls > 0]

    # Apply the Component analysis function
    (_, msk_cntd, _, centroids) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), cv2.CV_16U)

    # - For preserve the old labels
    msk_rpntd = np.zeros_like(msk_cntd, dtype=np.float32)
    # - Saves the labels to know if the labels was present
    lbl_cntd_roi_history = dict()
    for idx, lbl in enumerate(lbls):
        # TURN THE INSTANCE LABEL TO BINARY

        # - Copy the mask
        msk_bin = deepcopy(mask)

        # - Turn all the non-label pixels to 0
        msk_bin[msk_bin != lbl] = 0

        # - Turn all the label pixels to 1
        msk_bin[msk_bin > 0] = 1

        # - FIND THE CORRESPONDING CONNECTED COMPONENT IN THE CONNECTED COMPONENT LABEL
        msk_cntd_roi = msk_bin * msk_cntd
        lbls_cntd_roi, n_pxs = np.unique(msk_cntd_roi, return_counts=True)

        # REMOVE THE BACKGROUND LABEL

        # - Find the non-background pixel indices

        non_background_px = np.argwhere(lbls_cntd_roi > 0)

        # - Remove the background pixel label from the labels
        lbls_cntd_roi = lbls_cntd_roi[non_background_px]

        # - Remove the background pixel count from the pixel count
        n_pxs = n_pxs[non_background_px]

        # - Find the label with the maximum number of pixels in the ROI
        max_lbl_idx = np.argmax(n_pxs)

        # - Filter the labels with lower number of pixels in the ROI
        lbl_cntd_roi = lbls_cntd_roi[max_lbl_idx][0]

        # PAINT THE ROI
        msk_cntd_roi_bin = msk_cntd_roi / lbl_cntd_roi

        if lbl_cntd_roi not in lbl_cntd_roi_history.keys():
            # - If the color is new - paint it in this color and add it to the history
            msk_rpntd += msk_cntd_roi_bin * lbl

            # - Add the ROI label to history
            lbl_cntd_roi_history[lbl_cntd_roi] = lbl
        else:
            # - If the color was previously used - the cells were connected, so paint the ROI in the color previously used
            msk_rpntd += msk_cntd_roi_bin * lbl_cntd_roi_history.get(lbl_cntd_roi)

    return np.expand_dims(msk_rpntd, -1)


def build_data(args):
    files = args[0]
    bins = args[1]
    n_samples_in_range = args[2]
    output_dir = args[3]
    create_log = args[4]
    # - Create ranges from bins
    assert len(bins) >= 2, f'bins must include at least 2 values, but were provided only {len(bins)} value/s!'
    print(f'''
    =========================================================
    > Starting process {os.getpid()} on {len(files)} files
    =========================================================
    ''')
    bins_min, bins_max, bins_diff = np.min(bins), np.max(bins), bins[1] - bins[0]
    ranges = np.array(list(zip(np.arange(bins_min, bins_max + bins_diff, bins_diff), np.arange(bins_min + bins_diff, bins_max + 2 * bins_diff, bins_diff))))

    n_rngs = len(ranges)
    total_image_samples = n_rngs * n_samples_in_range

    mask_augs = augs.mask_augs(image_width=CROP_WIDTH)

    files_pbar = tqdm(files)
    file_run_times = np.array([])
    for img_fl, seg_fl in files_pbar:
        t_start = time.time()
        # - Load the gt mask
        gt_msk = load_image(image_file=str(seg_fl), add_channels=True)

        # - Create a dedicated dir for the current image augs
        msk_dir = output_dir / f'{get_parent_dir_name(path=img_fl)}_{get_file_name(path=img_fl)}'

        # - If the msk_dir for this file exists - skip it
        if msk_dir.is_dir():
            continue

        # - If the msk_dir for this file does not exist - create it
        os.makedirs(msk_dir)

        # - Create a counter array for the number of images in each range
        rng_cnt_arr = np.zeros(n_rngs, dtype=np.int16)

        # - Create a seg measure history array
        seg_msrs = np.array([])

        # - While not all the ranges for the current image have n_samples_in_range images
        rngs_done = sum(rng_cnt_arr == n_samples_in_range)
        cur_file_run_time = time.time() - t_start
        cur_rngs_pct = 100 * rngs_done / n_rngs

        while rngs_done < n_rngs:
            # - Augment the mask
            aug_res = mask_augs(image=gt_msk, mask=gt_msk)
            msk_aug = aug_res.get('mask')

            msk_aug = repaint_instance_segmentation(mask=msk_aug[..., 0])

            # - Calculate the seg measure
            seg_msr = calc_seg_score(gt_msk, msk_aug)[0]

            # - Get the index and the range where the seg measure falls in
            rng_min, rng_max, rng_idx = get_range(value=seg_msr, ranges=ranges)

            # - If the seg measure is in the desired range - save it
            if MIN_J < seg_msr < MAX_J and rng_cnt_arr[rng_idx] < n_samples_in_range:
                # - Create a file name
                f_name = float_2_str(seg_msr) + '.tif'

                # - Create the file to save the image, and make sure it is unique
                img_unique_fl = check_unique_file(file=msk_dir / f_name)

                # - Save the mask
                cv2.imwrite(str(img_unique_fl), msk_aug)

                # - Increase the images in range counter
                rng_cnt_arr[rng_idx] += 1

                # - Append the current seg measure to the seg measure history
                seg_msrs = np.append(seg_msrs, seg_msr)

            # - Update the number of ranges where there are n_samples_in_range samples
            rngs_done = sum(rng_cnt_arr == n_samples_in_range)

            # - Update current run time
            cur_file_run_time = time.time() - t_start

            # - Update the current samples number
            current_image_samples = sum(rng_cnt_arr)

            # - Update the number of percentage of the ranges where there are n_samples_in_range samples
            cur_rngs_pct = 100 * rngs_done / n_rngs

            # - Update the number of percentage of the ranges where there are n_samples_in_range samples
            cur_samples_pct = 100 * current_image_samples / total_image_samples

            # - Update the progress bar
            files_pbar.set_postfix(pid=f'{os.getpid()}', seg_measure=f'{seg_msr:.4f}', samples=f'{current_image_samples}/{total_image_samples} ({cur_samples_pct:.2f}%)', ranges=f'{rngs_done}/{n_rngs} ({cur_rngs_pct:.2f}%)')

            # - In order to avoid long runs we can break this loop if there are enough samples in most of the ranges
            if cur_file_run_time >= FILE_MAX_TIME and cur_samples_pct >= FILE_MIN_SAMPLES_PCT:
                break

        # - Add the file runtime to history
        file_run_times = np.append(file_run_times, cur_file_run_time)

        # - Create a log if necessary
        if create_log:
            # - Load the image
            img = load_image(image_file=str(img_fl), add_channels=True)

            # - Create a dedicated dir for the current image augs
            img_log_dir = output_dir / f'log/{get_parent_dir_name(path=img_fl)}_{get_file_name(path=img_fl)}'
            os.makedirs(img_log_dir)

            # - Plot the histogram of the seg measures
            plot_bins = np.append(bins, 1.0 + bins[2] - bins[1])  # <= for the histogram to include the 1.0
            plot_hist(
                data=seg_msrs,
                bins=plot_bins,
                save_file=img_log_dir / 'samples_distribution.png'
            )

            # - Plot the samples
            aug_msk_fls = [msk_dir / str(fl) for fl in os.listdir(msk_dir)]

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
                fig, ax = get_image_mask_figure(
                    image=img,
                    mask=gt_msk,
                    suptitle='GT (red) vs Pred (blue)',
                    title=f'Seg Measure = {str_2_float(seg_msr_str)}',
                    figsize=(20, 20),
                )

        print(f'''
        Process id: {os.getpid()}
        Stats for file \'{img_fl}\':
            - Run time: {get_runtime(seconds=time.time() - t_start)}
            - Mean for {len(file_run_times)} files: {get_runtime(seconds=file_run_times.mean())} +/- {get_runtime(seconds=file_run_times.std())}
        ''')

    print(f'''
    =============================
    > PID {os.getpid()} DONE
    =============================
    ''')


def clean_invalids(files, masks_root):
    invalid_files = []
    # - Load the gt mask
    for img_fl in files:
        fl_parent = get_parent_dir_name(path=img_fl)
        fl_name = get_file_name(path=img_fl)
        fl_msk_dir = masks_root / f'{fl_parent}_{fl_name}'
        if not fl_msk_dir.is_dir() or not os.listdir(masks_root / fl_msk_dir):
            invalid_files.append(img_fl)
            os.remove(img_fl)
    return invalid_files


def analyze_data(files: list, masks_root: pathlib.Path, n_samples: int = 5, clean_invalid_files: bool = False):
    msk_dir = masks_root / f'images'
    os.makedirs(msk_dir, exist_ok=True)

    seg_measures = np.array([])
    files_pbar = tqdm(files)
    for img_fl in files_pbar:
        img = load_image(image_file=img_fl)

        # - Load the gt mask
        fl_name = get_file_name(path=img_fl)
        fl_msk_dir = msk_dir / f'{get_parent_dir_name(path=img_fl)}_{fl_name}'

        # - If there is no directory for the current file, or there are too few examples - skip it
        if not fl_msk_dir.is_dir() or len(os.listdir(fl_msk_dir)) < 96:
            continue

        # - Get the masks which correspond to the current file
        msk_names = np.array(os.listdir(fl_msk_dir), dtype=object)
        n_msks = len(msk_names)

        # - Choose randomly n_samples masks
        rnd_idxs = np.random.randint(0, n_msks, n_samples)
        rnd_msk_names = msk_names[rnd_idxs]

        for rnd_msk_name in rnd_msk_names:
            rnd_msk_fl = fl_msk_dir / rnd_msk_name
            j_str = get_file_name(path=rnd_msk_fl)

            # - Strip the '-' character in case it was added due to overlap in generated J values
            j = str_2_float(str_val=j_str if '-' not in j_str else j_str[:j_str.index('-')])
            seg_measures = np.append(seg_measures, j)
            msk = load_image(image_file=str(rnd_msk_fl), add_channels=True)

            # - Plot the image the GT mask and the augmented mask
            fig, ax = get_image_mask_figure(
                image=img,
                mask=msk,
                suptitle='Image vs Mask (blue)',
                title=f'Seg Measure = {j}',
                figsize=(20, 20),
            )
            plt.close(fig)

    # - Plot the histogram of the seg measures
    plot_hist(
        data=seg_measures,
        bins=np.arange(0.0, 1.1, 0.1),
        save_file=masks_root / 'samples_distribution.png',
        overwrite=True
    )

    # - If true - the files which have no dir in the masks_dir will be deleted
    if clean_invalid_files:
        clean_invalids(files=files, masks_root=msk_dir)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The root directory in which to save the generated data')

    parser.add_argument('--regenerate', default=False, action='store_true', help=f'If to create a new data')

    parser.add_argument('--n_samples_in_range', type=int, default=N_SAMPLES_IN_RANGE, help='The total number of samples to generate')

    parser.add_argument('--max_cpus', type=int, default=MAX_CPUS, help='The maximal number of CPUs to use')

    parser.add_argument('--create_log', default=False, action='store_true', help=f'If to plot the samples images of the data')

    parser.add_argument('--analyze', default=False, action='store_true', help=f'If to run the analysis of the data')

    parser.add_argument('--clean', default=False, action='store_true', help=f'Deletes the files which have no dir in the masks image dir')

    parser.add_argument('--convert', default=False, action='store_true', help=f'Convert instance segmentation to categorical')

    return parser


# def get_files_under_dir(dir_path: pathlib.Path or str):
#     fls = []
#     for root, dirs, files in os.walk(dir_path, topdown=False):
#         for fl in files:
#             fls.append(fl)
#     fls = np.array(fls, dtype=object)
#     return fls
#

# def merge_categorical_masks(masks: np.ndarray):
#     msk = np.zeros(masks.shape[1:])
#     for cat_mks in masks:
#         msk += cat_mks
#
#     # - Fix the boundary if it was added several times from different cells
#     msk[msk > 2] = 2
#
#     return msk
#
#
# def split_instance_mask(instance_mask: np.ndarray):
#     """
#     Splits an instance mask into N binary masks
#     :param: instance_mask - mask where integers represent different objects
#     """
#     # - Ensure the multi-class label is populated with int values
#     inst_msk = instance_mask.astype(np.int16)
#
#     # - Find the classes
#     labels = np.unique(inst_msk)
#
#     # - Discard the background (0)
#     labels = labels[labels > 0]
#
#     # - Convert the ground truth mask to one-hot class masks
#     bin_masks = []
#     for lbl in labels:
#         bin_class_mask = deepcopy(inst_msk)
#         bin_class_mask[bin_class_mask != lbl] = 0
#         bin_class_mask[bin_class_mask > 0] = 1
#         bin_masks.append(bin_class_mask)
#
#     bin_masks = np.array(bin_masks)
#
#     return bin_masks


# def instance_2_categorical(mask):
#     # Shrinks the labels
#     inner_msk = grey_erosion(mask, size=2)
#
#     # Create the contur of the cells
#     contur_msk = mask - inner_msk
#     contur_msk[contur_msk > 0] = 2
#
#     # - Create the inner part of the cell
#     inner_msk[inner_msk > 0] = 1
#
#     # - Combine the inner and the contur masks to create the categorical mask with three classes, i.e., background 0, inner 1 and contur 2
#     cat_msk = inner_msk + contur_msk
#
#     return cat_msk
#

def save_categorical_mask(mask: np.ndarray, save_file: pathlib.Path, show: bool = False):
    # - Prepare the mask overlap image
    msk = np.zeros((*mask.shape[:-1], 3))

    # - Red channel - background
    bg_msk = deepcopy(mask)
    bg_msk[bg_msk == 1] = 2
    bg_msk[bg_msk == 0] = 1
    bg_msk[bg_msk != 1] = 0
    msk[..., 0] = bg_msk[..., 0]

    # - Green channel - inner cell
    inner_msk = deepcopy(mask)
    inner_msk[inner_msk != 1] = 0
    msk[..., 1] = inner_msk[..., 0]

    # - Blue channel - contur of the cell
    contur_msk = deepcopy(mask)
    contur_msk[contur_msk != 2] = 0
    msk[..., 2] = contur_msk[..., 0]

    plt.imshow(msk)

    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        plt.savefig(save_file)

    if show:
        plt.show()

    plt.close()


def plot_categorical_masks(mask_files: list, output_dir: pathlib.Path):
    assert_pathable(argument=output_dir, argument_name='output_dir')
    os.makedirs(output_dir, exist_ok=True)

    shp_dict = dict()
    for msk_fl in mask_files:
        inst_msk = load_image(image_file=str(msk_fl), add_channels=True)
        if inst_msk.shape not in shp_dict.keys():
            shp_dict[inst_msk.shape] = [msk_fl]
        else:
            shp_dict.get(inst_msk.shape).append(msk_fl)

        # - Apply the procedure for each cell separately
        bin_msks = split_instance_mask(instance_mask=inst_msk)
        cat_bin_msks = []
        for bin_msk in bin_msks:
            cat_msk = instance_2_categorical(masks=[bin_msk])[0]
            cat_bin_msks.append(cat_msk)
        cat_bin_msks = np.array(cat_bin_msks)
        cat_msk = merge_categorical_masks(masks=cat_bin_msks)
        save_categorical_mask(mask=cat_msk, save_file=output_dir / f'{get_parent_dir_name(path=msk_fl.parent)}_{get_file_name(path=msk_fl)}.png')


if __name__ == '__main__':

    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Scan the files in the data dir

    fls = scan_files(
        root_dir=DATA_DIR,
        seg_dir_postfix=SEG_DIR_POSTFIX,
        image_prefix=IMAGE_PREFIX,
        seg_prefix=SEG_PREFIX,
        seg_sub_dir=SEG_SUB_DIR
    )

    # - Build the data file
    output_dir = args.output_dir
    if not output_dir.is_dir() or args.regenerate:
        output_dir = args.output_dir / get_ts()
        os.makedirs(output_dir)

    if args.analyze:
        analyze_data(files=[fl_tpl[0] for fl_tpl in fls], masks_root=output_dir, n_samples=5)
    elif args.clean:
        clean_invalids(files=[fl_tpl[0] for fl_tpl in fls], masks_root=output_dir / 'masks')
    elif args.convert:
        plot_categorical_masks(mask_files=[fl[1] for fl in fls], output_dir=output_dir / 'categorical_masks_2')
    else:
        n_cpus = np.min([mp.cpu_count(), args.max_cpus])
        n_items = len(fls) // n_cpus
        splt_fls = get_split_data(data=fls, n_items=n_items)
        n_splits = len(splt_fls)
        args = list(
            zip(
                splt_fls,
                [np.arange(0.0, 1.0, 0.1)] * n_splits,
                [args.n_samples_in_range] * n_splits,
                [output_dir] * n_splits,
                [args.create_log] * n_splits
            )
        )
        with Pool(processes=n_cpus) as p:

            if fls:
                p.map(
                    build_data,
                    args
                )
            else:
                print(f'No files to generate data from !')
