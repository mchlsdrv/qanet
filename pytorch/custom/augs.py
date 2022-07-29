import albumentations as A
import albumentations.augmentations.transforms as tr
import numpy as np
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image
from configs.general_configs import (
    EROSION_SIZES,
    DILATION_SIZES,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)
from utils import aux_funcs, data_utils

__author__ = 'sidorov@post.bgu.ac.il'


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
    mask = np.transpose(mask, (1, 2, 0))
    seg_shp = mask.shape
    rnd_st = np.random.RandomState(None)

    alph = seg_shp[1] * 2
    sgm = seg_shp[1] * 0.15

    dx = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph
    dy = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph

    x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
    idxs = np.reshape(np.expand_dims(y, -1) + dy, (-1, 1)), np.reshape(np.expand_dims(x, -1) + dx, (-1, 1))

    mask = map_coordinates(np.reshape(mask, (-1, 1)), idxs, order=1, mode='reflect').reshape(seg_shp)

    mask = np.transpose(mask, (2, 0, 1))

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
                p=.75
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


def check_augs(data_dir, seg_dir_postfix='GT', image_prefix='t0', seg_prefix='man_seg0'):
    files = data_utils.scan_files(root_dir=data_dir, seg_dir_postfix=seg_dir_postfix, image_prefix=image_prefix, seg_prefix=seg_prefix)
    augs = mask_augs()

    imgs = []
    masks = []
    mask_augs = []
    jaccards = []
    for img_fl, seg_fl in files:
        img = np.array(Image.open(str(img_fl)).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(str(seg_fl)).convert('L'), dtype=np.float32)

        aug_res = augs(image=img, mask=mask)
        img_aug, mask_aug = aug_res.get('image'), aug_res.get('mask')

        jaccard = aux_funcs.calc_jaccard(mask, mask_aug)

        imgs.append(img)
        masks.append(mask)
        mask_augs.append(mask_aug)
        jaccards.append(jaccard)

    imgs = np.array(imgs)
    masks = np.array(masks)
    mask_augs = np.array(mask_augs)
    jaccards = np.array(jaccards)

    for idx in range(10):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(imgs[idx], cmap='gray')
        ax[1].imshow(masks[idx], cmap='gray')
        ax[2].imshow(mask_augs[idx], cmap='gray')
        fig.suptitle(f'Jaccard: {jaccards[idx]:.3f}')
