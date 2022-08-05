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

EROSION_SIZES = [5, 10, 15, 20, 25]
DILATION_SIZES = [5, 10, 15, 20, 25]
CLAHE_CLIP_LIMIT = 2
CLAHE_TILE_GRID_SIZE = 8
IMAGE_WIDTH = 419
IMAGE_HEIGHT = 419


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


def train_augs():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                p=1.
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
                A.SafeRotate(
                    limit=90,
                    interpolation=cv2.INTER_LANCZOS4,
                    p=0.5
                )
            ],
                p=.75
            ),
            # A.CLAHE(
            #     clip_limit=CLAHE_CLIP_LIMIT,
            #     tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
            #     p=1.
            # ),
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
                        p=.5
                    ),
                    tr.Lambda(
                        mask=random_dilation,
                        p=.5
                    ),
                    tr.Lambda(
                        mask=random_opening,
                        p=.5
                    ),
                    tr.Lambda(
                        mask=random_closing,
                        p=.5
                    ),
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        alpha_affine=50,
                        interpolation=cv2.INTER_LANCZOS4,
                        approximate=True,
                        same_dxdy=True,
                        p=0.5
                    ),
                ],
                p=.75
            ),
            A.ToFloat(p=1.),
        ]
    )


def val_augs():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                p=1.
            ),
            A.CLAHE(
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
                p=1.
            ),
            A.ToFloat(p=1.),
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
            A.CLAHE(
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
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
            A.CLAHE(
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
                p=1.
            ),
            A.ToFloat(p=1.),
        ]
    )
