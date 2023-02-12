import albumentations as A
import albumentations.augmentations.transforms as tr
import cv2
import numpy as np
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

__author__ = 'sidorov@post.bgu.ac.il'

# from global_configs.general_configs import CROP_HEIGHT, CROP_WIDTH

# ==============================================================================
# ==============================================================================
# * Parameters were found with the Optuna library, by minimizing the Mean
# Squared Error of the seg measure histogram produced by the augmentation and a
# vector of [0.5, 0.5, ..., 0.5]
# which represented ranges of 0.1, i.e., [0.0, 0.1, 0.2, ..., 0.9, 1.0], and
# meant to represent a balanced data, i.e., data in which there is as much as
# possible equal number of samples
# in each of the ranges above.
# ==============================================================================
# ==============================================================================

# - Erosion / dilation / opening / closing parameters
EROSION_SIZES = np.arange(4, 8)
DILATION_SIZES = np.arange(4, 8)

# - Elastic transform parameters
ALPHA_FACTOR = 0.1

SIGMA_FACTOR = 0.1

ALPHA_AFFINE_FACTOR = 0.1

# - Application probabilities
P_EROSION = 0.5
P_DILATION = 0.5
P_OPENING = 0.5
P_CLOSING = 0.5
P_ONE_OF = 1.0
P_ELASTIC = 0.7

# - CLAHE parameters
CLAHE_CLIP_LIMIT = 2
CLAHE_TILE_GRID_SIZE = 8


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(image.shape)


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
        ]
    )


def train_augs(crop_height: int, crop_width: int):
    return A.Compose(
        [
            A.CropNonEmptyMaskIfExists(
                height=crop_height,
                width=crop_width,
                p=1.
            ),
            A.OneOf(
                [
                    A.Flip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45,
                                       interpolation=cv2.INTER_LANCZOS4, p=0.5
                                       ),
                    A.GaussianBlur(p=.5),
                ],
                p=0.5
            ),
        ],
        # additional_targets={'mask0': 'mask'}
    )


def test_augs(crop_height: int, crop_width: int):
    return A.Compose(
        [
            A.CropNonEmptyMaskIfExists(
                height=crop_height,
                width=crop_width,
                p=1.
            ),
            # A.GaussianBlur(p=1),
        ]
    )


def inference_augs(crop_height: int, crop_width: int):
    return A.Compose(
        [
            A.CropNonEmptyMaskIfExists(
                height=crop_height,
                width=crop_width,
                p=1.
            ),
            # A.GaussianBlur(p=1),
        ]
    )
