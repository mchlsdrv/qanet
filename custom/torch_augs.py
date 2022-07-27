import time

import albumentations as A
import albumentations.augmentations.transforms as tr
import numpy as np
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from configs.torch_general_configs import (
    EROSION_SIZES,
    DILATION_SIZES,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE
)

__author__ = 'sidorov@post.bgu.ac.il'

from configs.torch_general_configs import IMAGE_WIDTH, IMAGE_HEIGHT


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


def elastic_transform(mask, **kwargs):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    t_strt = time.time()
    seg_shp = mask.shape
    rnd_st = np.random.RandomState(None)

    alph = seg_shp[1] * 2
    sgm = seg_shp[1] * 0.15

    dx = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph
    dy = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph

    x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
    idxs = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    mask = map_coordinates(mask, idxs, order=1, mode='reflect').reshape(seg_shp)
    return mask


def train_augs():
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
            A.OneOf([
                A.Flip(),
                A.RandomRotate90()
            ],
                p=.5
            ),
            ToTensorV2()
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
                    # tr.Lambda(
                    #     mask=elastic_transform,
                    #     p=.5
                    # ),
                ],
                p=1.
            ),
            A.ToFloat(p=1.),
            ToTensorV2()
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
            ToTensorV2()
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
            ToTensorV2()
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
            ToTensorV2()
        ]
    )
