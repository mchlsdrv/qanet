import numpy as np
import time
import albumentations as A
import albumentations.augmentations.transforms as tr
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from configs.general_configs import (
    IMAGE_SIZE,
    EROSION_SIZES,
    DILATION_SIZES, COARSE_DROPOUT_MAX_HOLES, COARSE_DROPOUT_MAX_HEIGHT, COARSE_DROPOUT_MAX_WIDTH, COARSE_DROPOUT_FILL_VALUE
)


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


def train_augmentations(configs: dict):
    return A.Compose(
        [
            # tr.CoarseDropout(
            #     max_holes=COARSE_DROPOUT_MAX_HOLES,
            #     max_height=COARSE_DROPOUT_MAX_HEIGHT,
            #     max_width=COARSE_DROPOUT_MAX_WIDTH,
            #     fill_value=COARSE_DROPOUT_FILL_VALUE,
            #     p=0.5
            # ),
            CropNonEmptyMaskIfExists(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.
            ),
            # A.ToGray(p=1.),
            A.CLAHE(
                clip_limit=configs.get('clahe')['clip_limit'],
                tile_grid_size=(configs.get('clahe')['tile_grid_size'], configs.get('clahe')['tile_grid_size']),
                p=1.
            ),
            A.ToFloat(p=1.),
            A.OneOf([
                A.Flip(),
                A.RandomRotate90(),
                # tr.GaussianBlur(
                #     blur_limit=(3, 7),
                #     sigma_limit=0,
                #     p=1.
                # ),
                # tr.GlassBlur(
                #     sigma=.7,
                #     max_delta=4,
                #     iterations=2,
                #     p=1.
                # ),
                # tr.GaussNoise(
                #     var_limit=(10., 50.),
                #     mean=0,
                #     p=1.
                # ),
                # tr.MultiplicativeNoise(
                #     multiplier=(.9, 1.1),
                #     elementwise=False,
                #     p=1.
                # ),
                # tr.RandomBrightnessContrast(
                #     brightness_limit=.2,
                #     contrast_limit=.2,
                #     p=1.
                # )
            ],
                p=.5
            ),
        ]
    )


def mask_deformations():
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
                    tr.Lambda(
                        mask=elastic_transform,
                        p=.5
                    ),
                ],
                p=1.
            )
        ]
    )


def validation_augmentations(configs: dict):
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.
            ),
            # A.ToGray(p=1.),
            A.CLAHE(
                clip_limit=configs.get('clahe')['clip_limit'],
                tile_grid_size=(configs.get('clahe')['tile_grid_size'], configs.get('clahe')['tile_grid_size']),
                p=1.
            ),
            A.ToFloat(p=1.),
        ]
    )


def inference_augmentations(configs: dict):
    return A.Compose(
        [
            A.Resize(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.
            ),
            # A.ToGray(p=1.),
            A.CLAHE(
                clip_limit=configs.get('clahe')['clip_limit'],
                tile_grid_size=(configs.get('clahe')['tile_grid_size'], configs.get('clahe')['tile_grid_size']),
                p=1.
            ),
            A.ToFloat(p=1.),
        ]
    )
