import numpy as np
import logging
import time
import cv2
import albumentations as A
import albumentations.augmentations.transforms as tr
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from scipy.ndimage import (
    grey_dilation,
    grey_erosion
)
from skimage.transform import (
    warp,
    AffineTransform
)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from configs.general_configs import (
    DEBUG_LEVEL,
    PROFILE,
    EROSION_SIZES,
    DILATION_SIZES,
    IMAGE_SIZE,
    EROSION,
    EROSION_SIZES,

    DILATION,
    DILATION_SIZES,

    OPENING,
    OPENING_SIZES,

    CLOSING,
    CLOSING_SIZES,

    # - Affine Transforms
    AFFINE,
    SCALE_RANGE,
    SHEER_RANGE,

    # - Elastic Transforms
    ELASTIC,
    SIGMA_RANGE,
    ALPHA_RANGE,
)
from utils.image_funcs import (
    add_channels_dim
)

def train_augmentations():
    return A.Compose(
        [
            tr.CoarseDropout(
                max_holes=8,
                max_height=30,
                max_width=30,
                fill_value=0,
                p=1
            ),
            CropNonEmptyMaskIfExists(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.
            ),
            A.OneOf([
                A.Flip(),
                A.RandomRotate90(),
            ], p=.5),
            A.OneOf([
                tr.GaussianBlur(
                    blur_limit=(3, 7),
                    sigma_limit=0,
                    p=1.
                ),
                tr.GlassBlur(
                    sigma=.7,
                    max_delta=4,
                    iterations=2,
                    p=1.
                ),
                tr.GaussNoise(
                    var_limit=(10., 50.),
                    mean=0,
                    p=1.
                ),
                tr.MultiplicativeNoise(
                    multiplier=(.9, 1.1),
                    elementwise=False,
                    p=1.
                ),
                tr.RandomBrightnessContrast(
                    brightness_limit=.2,
                    contrast_limit=.2,
                    p=1.
                )
            ], p=.5),
        ]
    )


def validation_augmentations():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                p=1.
            ),
        ]
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


def affine_transform(mask, **kwargs):
    scl = np.random.uniform(*SCALE_RANGE, 2)
    tform = AffineTransform(scale=scl + 1, shear=np.random.uniform(*SHEER_RANGE))
    mask = warp(mask, tform.inverse, output_shape=mask.shape)

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


def mask_augmentations():
    return A.Compose(
        [
            A.OneOf(
                [
                    tr.Lambda(
                        mask=random_erosion,
                        p=1.
                    ),
                    tr.Lambda(
                        mask=random_dilation,
                        p=1.
                    ),
                    tr.Lambda(
                        mask=random_opening,
                        p=1.
                    ),
                    tr.Lambda(
                        mask=random_closing,
                        p=1.
                    ),
                ],
                p=.5
            ),
            tr.Lambda(
                mask=affine_transform,
                p=.5
            ),
            tr.Lambda(
                mask=elastic_transform,
                p=.5
            ),
        ]
    )




# def get_random_left_corner(image, crop_width: int, crop_height: int, logger: logging.Logger = None):
#     image_width, image_height = image.shape[0], image.shape[1]
#
#     img = image
#     # - If the image is in BGR format
#     if len(img.shape) > 2 and img.shape[-1] > 1:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         blr = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
#         ret, img = cv2.threshold(blr, 200, 255, cv2.THRESH_BINARY_INV)
#
#     # - If the image is a multiclass grayscale image
#     elif len(np.unique(img)) > 2:
#         idxs = np.argwhere(img > 0)
#         x, y = idxs[:, 0], idxs[:, 1]
#         img[(x, y)] = 1
#
#     # - Images should be of type UINT8
#     img = img.astype(np.uint8)
#
#     # - Find the contours in the image
#     contours, centroids = get_contours(image=img)
#
#     # - If theres at least one centroid
#     if centroids:
#         # -*- Choose a random centroid
#         # rnd_cntr = centroids[np.random.choice(np.arange(len(centroids)))]
#         rnd_cntr_x, rnd_cntr_y = centroids[np.random.choice(np.arange(len(centroids)))]
#
#         # -*- Find the minimum and the maximum between which we can randomly choose the x and y coordinates,
#         # so that the centroid will fall inside the square to the top-left of the chosen centroid
#         # --> Minimum
#         crop_width = crop_width // 2
#         crop_height = crop_height // 2
#
#         x_min, y_min = rnd_cntr_x - crop_width if rnd_cntr_x - crop_width >= 0 else 0, rnd_cntr_y - crop_height if rnd_cntr_y - crop_height >= 0 else 0
#
#         # --> Maximum
#         x_residual, y_residual = (rnd_cntr_x + crop_width) - image_width, (rnd_cntr_y + crop_height) - image_height
#         x_max, y_max = rnd_cntr_x if x_residual <= 0 else rnd_cntr_x - x_residual, rnd_cntr_y if y_residual <= 0 else rnd_cntr_y - y_residual
#
#         rnd_x, rnd_y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)
#
#         if rnd_x + IMAGE_SIZE > image_height:
#             rnd_x = rnd_x - (rnd_x + IMAGE_SIZE - image_height)
#
#         if rnd_y + IMAGE_SIZE > image_height:
#             rnd_y = rnd_y - (rnd_y + IMAGE_SIZE - image_height)
#
#     else:
#         info_log(logger=logger, message=f'\nCould not crop the image based on contours! Falling back to the random crops!')
#         rnd_x, rnd_y = np.random.randint(0, image_width - crop_width), np.random.randint(0, image_height - crop_height)
#
#     return rnd_x, rnd_y
#
#
# def random_crop(image, segmentation, logger: logging.Logger = None):
#     t_strt = time.time()
#
#     x, y = get_random_left_corner(image=segmentation, crop_width=IMAGE_SIZE, crop_height=IMAGE_SIZE)
#
#     # 2) Randomly crop the image and the label
#     img_crp = get_crop(image=image, x=x, y=y, crop_shape=(IMAGE_SIZE, IMAGE_SIZE))
#     seg_crp = get_crop(image=segmentation, x=x, y=y, crop_shape=(IMAGE_SIZE, IMAGE_SIZE))
#
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Random crop took {get_runtime(seconds=time.time() - t_strt)}')
#
#     return img_crp, seg_crp
#
#
# def random_erosion(segmentation, kernel=None, logger: logging.Logger = None):
#     # Shrinks the labels
#     t_strt = time.time()
#     krnl = kernel
#     if krnl is None:
#         krnl = np.random.choice(EROSION_SIZES)
#     seg = grey_erosion(segmentation, size=krnl)
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Erosion took {get_runtime(seconds=time.time() - t_strt)}')
#     return seg
#
#
# def random_dilation(segmentation, kernel=None, logger: logging.Logger = None):
#     # "Fattens" the cell label
#     t_strt = time.time()
#     krnl = kernel
#     if krnl is None:
#         krnl = np.random.choice(DILATION_SIZES)
#     seg = grey_dilation(segmentation, size=krnl)
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Dilation took {get_runtime(seconds=time.time() - t_strt)}')
#     return seg
#
#
# def random_opening(segmentation, logger: logging.Logger = None):
#     t_strt = time.time()
#     # Connected labels are brought apart
#     krnl = np.random.choice(OPENING_SIZES)
#
#     seg = random_dilation(random_erosion(segmentation, kernel=krnl), kernel=krnl)
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Opening took {get_runtime(seconds=time.time() - t_strt)}')
#     return seg
#
#
# def random_closing(segmentation, logger: logging.Logger = None):
#     t_strt = time.time()
#     # Disconnected labels are brought together
#     krnl = np.random.choice(CLOSING_SIZES)
#     seg = random_erosion(random_dilation(segmentation, kernel=krnl), kernel=krnl)
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Closing took {get_runtime(seconds=time.time() - t_strt)}')
#     return seg
#
#
# def affine_transform(segmentation, logger: logging.Logger = None):
#     t_strt = time.time()
#     scl = np.random.uniform(*SCALE_RANGE, 2)
#     tform = AffineTransform(scale=scl + 1, shear=np.random.uniform(*SHEER_RANGE))
#     seg = warp(segmentation, tform.inverse, output_shape=segmentation.shape)
#
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Affine transform took {get_runtime(seconds=time.time() - t_strt)}')
#     return seg
#
#
# def elastic_transform(segmentation, logger: logging.Logger = None):
#     """Elastic deformation of images as described in [Simard2003]_.
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#     Convolutional Neural Networks applied to Visual Document Analysis", in
#     Proc. of the International Conference on Document Analysis and
#     Recognition, 2003.
#     """
#     t_strt = time.time()
#     seg_shp = segmentation.shape
#     rnd_st = np.random.RandomState(None)
#
#     alph = seg_shp[1] * 2
#     sgm = seg_shp[1] * 0.15
#
#     dx = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph
#     dy = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph
#
#     x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
#     idxs = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
#
#     seg = map_coordinates(segmentation, idxs, order=1, mode='reflect').reshape(seg_shp)
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'Elastic transform took {get_runtime(seconds=time.time() - t_strt)}')
#
#     return seg
#
#
# # def augment(image, segmentation, rotation: bool = True, affine: bool = True, erosion: bool = True, dilation: bool = True, opening: bool = True, closing: bool = True, elastic: bool = True, logger: logging.Logger = None):
# # def spoil_segmentation(image, segmentation, logger: logging.Logger = None):
# def spoil_segmentation(segmentation, logger: logging.Logger = None):
#     t_strt = time.time()
#
#     # Custom
#     spoiled_seg = segmentation
#     #  1) Non-ridged (Affine)
#     if AFFINE and np.random.random() > .5:
#         spoiled_seg = affine_transform(
#             segmentation=segmentation,
#             logger=logger
#         )
#
#     #  2) Morphological
#     if EROSION and np.random.random() > .5:
#         spoiled_seg = random_erosion(
#             segmentation=spoiled_seg,
#             logger=logger
#         )
#     elif OPENING and np.random.random() > .5:
#         spoiled_seg = random_opening(
#             segmentation=spoiled_seg,
#             logger=logger
#         )
#
#     elif DILATION and np.random.random() > .5:
#         spoiled_seg = random_dilation(
#             segmentation=spoiled_seg,
#             logger=logger
#         )
#     elif CLOSING and np.random.random() > .5:
#         spoiled_seg = random_closing(
#             segmentation=spoiled_seg,
#             logger=logger
#         )
#
#     #  3) Elastic
#     if ELASTIC and np.random.random() > .5:
#         spoiled_seg = elastic_transform(
#             segmentation=spoiled_seg,
#             logger=logger
#         )
#
#     # if len(img.shape) < 3:
#     #     img = add_channels_dim(
#     #         image=img
#     #     )
#     # if len(seg.shape) < 3:
#     #     seg = add_channels_dim(
#     #         image=seg
#     #     )
#     if len(spoiled_seg.shape) < 3:
#         spoiled_seg = add_channels_dim(
#             image=spoiled_seg
#         )
#
#     if PROFILE and DEBUG_LEVEL > 2:
#         info_log(logger=logger, message=f'All augmentations took {get_runtime(seconds=time.time() - t_strt)}')
#
#     return spoiled_seg
