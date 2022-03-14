import time
import numpy as np
import logging
import time
import cv2
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
    OPENING_SIZES,
    CLOSING_SIZES,
    SCALE_RANGE,
    SHEER_RANGE,
    SIGMA_RANGE,
    ALPHA_RANGE,
    CROP_SIZE,
    NON_EMPTY_CROPS,
    NON_EMPTY_CROP_THRESHOLD,
    MAX_EMPTY_CROPS,
)
from utils.image_utils.image_aux import (
    get_crop,
    add_channels_dim
)

from utils.general_utils.aux_funcs import (
    get_runtime,
    info_log,
    err_log
)


def random_rotation(image: np.ndarray, segmentation: np.ndarray, logger: logging.Logger = None) -> (np.ndarray, np.ndarray):
    t_strt = time.time()
    dgrs = np.random.randint(-180, 180)
    # Rotates the image by degrees
    img_shp = image.shape
    h, w = img_shp[0], img_shp[1]

    # - Represents the point around which the image will be rotated
    cX, cY = w // 2, h // 2
    rot_pt = (cX, cY)

    # - Configures the rotation matrix, which is multiplied by the image to create the rotation
    # > The first argument represents the point around which the rotation happens
    # > The second argument represents the degrees by which to rotate the image
    # > The last argument represents the scaling factor of the output image
    M = cv2.getRotationMatrix2D(rot_pt, dgrs, 1.)

    # - Performs the actual rotation
    rot_img = cv2.warpAffine(image, M, img_shp[:-1])
    rot_seg = cv2.warpAffine(segmentation, M, img_shp[:-1])

    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Random rotation took {get_runtime(seconds=time.time() - t_strt)}')
    return rot_img, rot_seg


def random_crop(image, segmentation, non_empty: bool = True, logger: logging.Logger = None):
    t_strt = time.time()
    h, w = CROP_SIZE, CROP_SIZE
    x = np.random.randint(0, image.shape[0] - h)

    # - y coordinate
    y = np.random.randint(0, image.shape[1] - w)

    # 2) Randomly crop the image and the label
    img_crp = get_crop(image=image, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))
    seg_crp = get_crop(image=segmentation, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))

    if non_empty and seg_crp.sum() < NON_EMPTY_CROP_THRESHOLD:
        # 1) Produce random x, y coordinates with size of crop_shape
        for try_idx in range(MAX_EMPTY_CROPS):
            # - x coordinate
            x = np.random.randint(0, image.shape[0] - h)

            # - y coordinate
            y = np.random.randint(0, image.shape[1] - w)

            # 2) Randomly crop the image and the label
            img_crp = get_crop(image=image, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))
            seg_crp = get_crop(image=segmentation, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))

            # 3) Check if one of the following happens:
            # - the crop contains some foreground
            if seg_crp.sum() > NON_EMPTY_CROP_THRESHOLD:
                break
            if DEBUG_LEVEL > 1:
                info_log(logger=logger, message=f'The crops\' sum is {seg_crp.sum()} < {NON_EMPTY_CROP_THRESHOLD}. Trying to acquire another crop (try #{try_idx})...')

    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Random crop took {get_runtime(seconds=time.time() - t_strt)}')

    return img_crp, seg_crp


def random_erosion(segmentation, kernel=None, logger: logging.Logger = None):
    # Shrinks the labels
    t_strt = time.time()
    krnl = kernel
    if krnl is None:
        krnl = np.random.choice(EROSION_SIZES)
    seg = grey_erosion(segmentation, size=krnl)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Erosion took {get_runtime(seconds=time.time() - t_strt)}')
    return seg


def random_dilation(segmentation, kernel=None, logger: logging.Logger = None):
    # "Fattens" the cell label
    t_strt = time.time()
    krnl = kernel
    if krnl is None:
        krnl = np.random.choice(DILATION_SIZES)
    seg = grey_dilation(segmentation, size=krnl)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Dilation took {get_runtime(seconds=time.time() - t_strt)}')
    return seg


def random_opening(segmentation, logger: logging.Logger = None):
    t_strt = time.time()
    # Connected labels are brought apart
    krnl = np.random.choice(OPENING_SIZES)

    seg = random_dilation(random_erosion(segmentation, kernel=krnl), kernel=krnl)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Opening took {get_runtime(seconds=time.time() - t_strt)}')
    return seg


def random_closing(segmentation, logger: logging.Logger = None):
    t_strt = time.time()
    # Disconnected labels are brought together
    krnl = np.random.choice(CLOSING_SIZES)
    seg = random_erosion(random_dilation(segmentation, kernel=krnl), kernel=krnl)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Closing took {get_runtime(seconds=time.time() - t_strt)}')
    return seg


def affine_transform(segmentation, logger: logging.Logger = None):
    t_strt = time.time()
    scl = np.random.uniform(*SCALE_RANGE, 2)
    tform = AffineTransform(scale=scl + 1, shear=np.random.uniform(*SHEER_RANGE))
    seg = warp(segmentation, tform.inverse, output_shape=segmentation.shape)

    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Affine transform took {get_runtime(seconds=time.time() - t_strt)}')
    return seg


def elastic_transform(segmentation, logger: logging.Logger = None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    t_strt = time.time()
    seg_shp = segmentation.shape
    rnd_st = np.random.RandomState(None)

    sgm = np.random.uniform(*SIGMA_RANGE)  #1, 8)
    alph = np.random.uniform(*ALPHA_RANGE)  #50, 100)

    dx = gaussian_filter(rnd_st.rand(*seg_shp) * 2 - 1, sgm, mode='constant', cval=0) * alph
    dy = gaussian_filter(rnd_st.rand(*seg_shp) * 2 - 1, sgm, mode='constant', cval=0) * alph

    x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
    idxs = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    seg = map_coordinates(segmentation, idxs, order=1, mode='reflect').reshape(seg_shp)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Elastic transform took {get_runtime(seconds=time.time() - t_strt)}')

    return seg


def augment(image, segmentation, non_empty_crops: bool = True, rotation: bool = True, affine: bool = True, erosion: bool = True, dilation: bool = True, opening: bool = True, closing: bool = True, elastic: bool = True, logger: logging.Logger = None):
    t_strt = time.time()

    img, seg = image[:, :, 0], segmentation[:, :, 0]
    # I. Image + segmentation
    #  1) Random rotation (whole of the image)
    if rotation:
        img, seg = random_rotation(
            image=image,
            segmentation=segmentation,
            logger=logger
        )

    #  2) Random crop
    img, seg = random_crop(
        image=img,
        segmentation=seg,
        non_empty=non_empty_crops,
        logger=logger
    )

    # II. Segmentation only
    spoiled_seg = seg
    #  1) Non-ridged (Affine)
    if affine and np.random.random() >= .5:
        spoiled_seg = affine_transform(
            segmentation=seg,
            logger=logger
        )

    #  2) Morphological
    if erosion and np.random.random() > .5:
        spoiled_seg = random_erosion(
            segmentation=spoiled_seg,
            logger=logger
        )

    if dilation and np.random.random() > .5:
        spoiled_seg = random_dilation(
            segmentation=spoiled_seg,
            logger=logger
        )

    if opening and np.random.random() > .5:
        spoiled_seg = random_opening(
            segmentation=spoiled_seg,
            logger=logger
        )

    if closing and np.random.random() > .5:
        spoiled_seg = random_closing(
            segmentation=spoiled_seg,
            logger=logger
        )

    #  3) Elastic
    if elastic and np.random.random() >= .5:
        spoiled_seg = elastic_transform(
            segmentation=spoiled_seg,
            logger=logger
        )

    if len(img.shape) < 3:
        img = add_channels_dim(
            image=img
        )
    if len(seg.shape) < 3:
        seg = add_channels_dim(
            image=seg
        )
    if len(spoiled_seg.shape) < 3:
        spoiled_seg = add_channels_dim(
            image=spoiled_seg
        )

    if PROFILE and DEBUG_LEVEL > 2:
            info_log(logger=logger, message=f'All augmentations took {get_runtime(seconds=time.time() - t_strt)}')

    return img, seg, spoiled_seg

