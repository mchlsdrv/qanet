import albumentations as A
import albumentations.augmentations.transforms as tr
import cv2
import numpy as np
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)

__author__ = 'sidorov@post.bgu.ac.il'

# - Image crop parameters
IMAGE_WIDTH = 419
IMAGE_HEIGHT = 419
# ========================================================================================================================================================================================
# ========================================================================================================================================================================================
# * Parameters were found with the Optuna library, by minimizing the Mean Squared Error of the seg measure histogram produced by the augmentation and a vector of [0.5, 0.5, ..., 0.5]
# which represented ranges of 0.1, i.e., [0.0, 0.1, 0.2, ..., 0.9, 1.0], and meant to represent a balanced data, i.e., data in which there is as much as possible equal number of samples
# in each of the ranges above.
# ========================================================================================================================================================================================
# ========================================================================================================================================================================================

# - Erosion / dilation / opening / closing parameters
EROSION_SIZES = [1, 3, 5, 7]
DILATION_SIZES = [1, 3, 5, 7, 9, 11, 13, 15, 17]

# - Elastic transform parameters
ALPHA_FACTOR = 1.2531
ALPHA = ALPHA_FACTOR * IMAGE_WIDTH

SIGMA_FACTOR = 0.8204
SIGMA = SIGMA_FACTOR * IMAGE_WIDTH

ALPHA_AFFINE_FACTOR = 0.0143
ALPHA_AFFINE = ALPHA_AFFINE_FACTOR * IMAGE_WIDTH

# - Application probabilities
P_EROSION = 0.675
P_DILATION = 0.259
P_OPENING = 0.383
P_CLOSING = 0.820
P_ONE_OF = 0.907
P_ELASTIC = 0.5

# - CLAHE parameters
CLAHE_CLIP_LIMIT = 2
CLAHE_TILE_GRID_SIZE = 8


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
    # Disconnects labels
    mask = random_dilation(random_erosion(mask))
    return mask


def random_closing(mask, **kwargs):
    # Disconnected labels are brought together
    mask = random_erosion(random_dilation(mask))
    return mask


def image_mask_augs():
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=(0.5, 1.5),
                p=0.5
            ),
            A.OneOf([
                A.Flip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    interpolation=cv2.INTER_LANCZOS4,
                    p=0.5
                ),
                ], p=0.5
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


def mask_augs():
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


def test_augs():
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


def inference_augs():
    return A.Compose(
        [
            A.Resize(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                p=1.
            ),
            A.ToFloat(p=1.),
        ]
    )
