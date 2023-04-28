from functools import partial

import albumentations as A
import albumentations.augmentations.transforms as tr
import numpy as np
from scipy.ndimage import (
    grey_dilation,
    grey_erosion,
    grey_closing,
    grey_opening,
)

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

__author__ = 'sidorov@post.bgu.ac.il'

from configs.general_configs import EPSILON

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
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(image.shape)


# noinspection PyUnusedLocal
def random_erosion(mask, **kwargs):
    # Shrinks the labels
    krnl = np.random.choice(EROSION_SIZES)
    mask = grey_erosion(mask, size=krnl)
    return mask


# noinspection PyUnusedLocal
def random_dilation(mask, **kwargs):
    # "Fattens" the cell label
    krnl = np.random.choice(DILATION_SIZES)
    mask = grey_dilation(mask, size=krnl)
    return mask


# noinspection PyUnusedLocal
def random_opening(mask, **kwargs):
    # Disconnects labels
    krnl = np.random.choice(EROSION_SIZES)
    mask = grey_opening(mask, size=krnl)
    return mask


# noinspection PyUnusedLocal
def random_closing(mask, **kwargs):
    # Disconnected labels are brought together
    krnl = np.random.choice(DILATION_SIZES)
    mask = grey_closing(mask, size=krnl)
    return mask


def standardize_image(image, **kwargs):
    return (image - image.mean()) / (image.std() + EPSILON)


def normalize_image(image, **kwargs):
    return (image - image.min()) / (image.max() - image.min() + EPSILON)


def image_clip_values(image, max_val, **kwargs):
    image[image > max_val] = max_val
    return image


def image_2_float(image, max_val, **kwargs):
    return image / max_val if max_val != 0 else image


def random_brightness(image, **kwargs):
    # - Random brightness delta plus/minus 10% of maximum value
    dlt = (np.random.rand() - 0.5) * 0.2 * image.max()
    return image + dlt


def random_contrast(image, **kwargs):
    # - Random contrast plus/minus 50%
    fctr = np.random.rand() + 0.5
    img_mean = image.mean()
    return (image - img_mean) * fctr + img_mean


def image_transforms():
    return A.Compose(
        [
            tr.Lambda(
                mask=partial(image_2_float, max_val=255),
                p=1.0
            ),
            tr.Lambda(
                mask=standardize_image,
                p=1.0
            ),
        ],
    )


def train_augs(crop_height: int, crop_width: int):
    return A.Compose(
        [
            A.CropNonEmptyMaskIfExists(
                height=crop_height,
                width=crop_width,
                p=1.
            ),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ],
    )


def train_image_augs():
    return A.Compose(
        [
            A.OneOf([
                tr.Lambda(
                    mask=random_contrast,
                    p=0.5
                ),
                tr.Lambda(
                    mask=random_brightness,
                    p=0.5
                ),
                A.PixelDropout(p=0.5),
                A.JpegCompression(p=0.5),
                A.GaussNoise(p=0.5)
            ], p=0.5
            )
        ],
    )


def train_mask_augs():
    return A.Compose(
        [
            A.OneOf(
                [
                    tr.Lambda(
                        mask=random_erosion,
                        p=1.0
                    ),
                    tr.Lambda(
                        mask=random_dilation,
                        p=1.0
                    ),
                    tr.Lambda(
                        mask=random_opening,
                        p=1.0
                    ),
                    tr.Lambda(
                        mask=random_closing,
                        p=1.0
                    ),
                ],
                p=0.6
            ),
        ]
    )
