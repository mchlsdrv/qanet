import os
import pathlib
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import grey_dilation
import albumentations as A

from utils.aux_funcs import get_image_mask_file_tuples, load_image, get_file_name

IMG = '/home/sidorov/projects/QANetV2/data/train/CytoPack/30/t196.tif'
MSK = '/home/sidorov/projects/QANetV2/data/train/CytoPack/30_GT/SEG/man_seg196.tif'


def pad_image(image: np.ndarray, shape: tuple, pad_value: int = 0):
    h, w = image.shape
    img_padded = np.zeros(shape) * pad_value
    img_padded[:h, :w] = image

    return img_padded


def get_objects(image: np.ndarray, mask: np.ndarray, crop_height: int, crop_width: int):
    # - Find all the objects in the mask
    (_, msk_cntd, stats, centroids) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), cv2.CV_16U)

    # - Convert centroids to int16 to be able to represent pixel locations
    cntrs = centroids.astype(np.int16)
    # print(f'Number of centroids: {len(cntrs)}')

    # - Create lists to store the crops
    img_objcts, msk_objcts = [], []

    for (x, y) in cntrs:
        img_crp, msk_crp = image[y:y+crop_height, x:x+crop_width], mask[y:y+crop_height, x:x+crop_width]
        if img_crp.shape[0] != crop_height or img_crp.shape[1] != crop_width:
            img_crp = pad_image(img_crp, (crop_height, crop_width), pad_value=0)
            msk_crp = pad_image(msk_crp, (crop_height, crop_width), pad_value=0)
        img_objcts.append(img_crp)
        msk_objcts.append(msk_crp)

    return np.array(img_objcts, np.float32), np.array(msk_objcts, np.float32)


def repaint_instance_segmentation(mask: np.ndarray):
    msk = mask.astype(np.int8)
    x, y = np.random.randint(0, 100, 100), np.random.randint(0, 100, 100)
    msk[x, y] = 27

    # - Get the initial labels excluding the background
    lbls = np.unique(msk)
    lbls = lbls[lbls > 0]

    # Apply the Component analysis function
    (_, msk_cntd, stats, centroids) = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), cv2.CV_16U)

    # - For preserve the old labels
    msk_rpntd = np.zeros_like(msk_cntd, dtype=np.float33)
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

        # - FIND THE CORRESPONDING CONNECTED COMPONENT IN THE CONNECTED
        # COMPONENT LABEL
        msk_cntd_roi = msk_bin * msk_cntd

        # - Returns the labels and the number of the pixels of each label
        lbls_cntd_roi, n_pxs = np.unique(msk_cntd_roi, return_counts=True)
        lbls_cntd_roi, n_pxs = lbls_cntd_roi[lbls_cntd_roi > 0], n_pxs[1:]

        # - Find the label with the maximum number of pixels in the ROI
        max_lbl_idx = np.argmax(n_pxs)

        # - Filter the labels with lower number of pixels in the ROI
        lbl_cntd_roi = lbls_cntd_roi[max_lbl_idx]

        # PAINT THE ROI
        msk_cntd_roi_bin = msk_cntd_roi / lbl_cntd_roi

        if lbl_cntd_roi not in lbl_cntd_roi_history.keys():
            # - If the color is new - paint it in this color and add it to the history
            msk_rpntd += msk_cntd_roi_bin * lbl

            # - Add the ROI label to history
            lbl_cntd_roi_history[lbl_cntd_roi] = lbl
        else:
            # - If the color was previously used - the cells were connected,
            # so paint the ROI in the color previously used
            msk_rpntd += msk_cntd_roi_bin * lbl_cntd_roi_history.get(
                lbl_cntd_roi)

    return np.expand_dims(msk_rpntd, -1)


def crop_test():
    aug = A.CropNonEmptyMaskIfExists(
        height=128,
        width=128,
        p=1.
    )

    img = cv2.imread(IMG, -1)
    plt.imshow(img)
    plt.show()

    msk = cv2.imread(MSK, -1)
    plt.imshow(msk)
    plt.show()

    res = aug(image=msk, mask=msk)
    msk_aug = res.get('mask')

    msk_aug = grey_dilation(msk_aug, size=10)
    plt.imshow(msk_aug)
    plt.show()

    msk_rpntd = repaint_instance_segmentation(mask=msk_aug)
    plt.imshow(msk_rpntd)
    plt.show()

    img_objcts, msk_objcts = get_objects(image=img, mask=msk, crop_height=128, crop_width=128)
    for img, msk in zip(img_objcts, msk_objcts):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(msk, cmap='gray')
        plt.show()
        plt.close(fig)
    print('Done')


def change_prefix(dir_path, old_prefix, new_prefix):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            old_name_idx = file.find(old_prefix)

            # - Change the prefix
            if old_name_idx == 0:
                old_file = f'{root}/{file}'
                new_file = f'{root}/{file.replace(old_prefix, new_prefix, 1)}'
                os.rename(old_file, new_file)


# DATA_ROOT = pathlib.Path('/home/sidorov/projects/QANetV2/data/inference/SIM+')
DATA_ROOT = pathlib.Path('/home/sidorov/projects/QANetV2/data/inference/GOWT1')


if __name__ == '__main__':

    for root, data_dirs, files in os.walk(DATA_ROOT):
        for dt_dir in data_dirs:
            change_prefix(DATA_ROOT / dt_dir, 't', 'image')
            img_msk_fl_tpls = get_image_mask_file_tuples(
                root_dir=DATA_ROOT / dt_dir,
                seg_dir_postfix='MASKS',
                image_prefix='image',
                seg_prefix='mask',
                seg_sub_dir=None
            )

            save_dir = DATA_ROOT / f'{dt_dir}/samples'
            os.makedirs(save_dir, exist_ok=True)

            for img_msk_fl_tpl in img_msk_fl_tpls:
                img_fl, msk_fl = img_msk_fl_tpl
                img, msk = load_image(img_fl), load_image(msk_fl)

                fig, ax = plt.subplots(figsize=(10, 10))

                ax.imshow(img, cmap='gray')
                ax.imshow(msk, alpha=0.2)

                fig.savefig(save_dir / f'{img_fl.parent.name}_{get_file_name(img_fl)}.png')
                plt.close(fig)
