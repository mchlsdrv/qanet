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
    CROP_SIZE,
)
from utils.image_utils.image_aux import (
    get_crop,
    add_channels_dim
)

from utils.aux_funcs import (
    get_runtime,
    info_log,
    err_log,
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


def get_random_left_corner(image, crop_width: int, crop_height: int, logger: logging.Logger = None):
    image_width, image_height = image.shape[0], image.shape[1]

    img = image
    # - If the image is in BGR format
    if len(img.shape) > 2 and img.shape[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blr = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        ret, img = cv2.threshold(blr, 200, 255, cv2.THRESH_BINARY_INV)

    # - If the image is a multiclass grayscale image
    elif len(np.unique(img)) > 2:
        idxs = np.argwhere(img > 0)
        x, y = idxs[:, 0], idxs[:, 1]
        img[(x, y)] = 1

    # - Images should be of type UINT8
    img = img.astype(np.uint8)

    # - Find the contours
    contours, hierarchies = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # - Find the centroids of the contours
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

    # - Choose a random centroid
    try:
        rnd_cntr = centroids[np.random.choice(np.arange(len(centroids)))]
        rnd_cntr_x, rnd_cntr_y = rnd_cntr

        # - Find the minimum and the maximum between which we can randomly choose the x and y coordinates, so that the centroid will fall inside the square
        # --> Minimum
        x_min, y_min = rnd_cntr_x - crop_width if rnd_cntr_x - crop_width >= 0 else 0, rnd_cntr_y - crop_height if rnd_cntr_y - crop_height >= 0 else 0

        # --> Maximum
        x_residual, y_residual = (rnd_cntr_x + crop_width) - image_width, (rnd_cntr_y + crop_height) - image_height
        x_max, y_max = rnd_cntr_x if x_residual <= 0 else rnd_cntr_x - x_residual, rnd_cntr_y if y_residual <= 0 else rnd_cntr_y - y_residual

        rnd_x, rnd_y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

    except ValueError as err:
        err_log(logger=logger, message=f'\nCould not crop the image based on contours! Falling back to the basic crops!')
        rnd_x, rnd_y = np.random.randint(0, image_width - crop_width), np.random.randint(0, image_height - crop_height)

    return rnd_x, rnd_y


def random_crop(image, segmentation, logger: logging.Logger = None):
    t_strt = time.time()

    x, y = get_random_left_corner(image=segmentation, crop_width=CROP_SIZE, crop_height=CROP_SIZE)

    # 2) Randomly crop the image and the label
    img_crp = get_crop(image=image, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))
    seg_crp = get_crop(image=segmentation, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))

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

    alph = seg_shp[1] * 2
    sgm = seg_shp[1] * 0.15

    dx = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph
    dy = gaussian_filter((rnd_st.rand(*seg_shp) * 2 - 1), sgm) * alph

    x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
    idxs = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    seg = map_coordinates(segmentation, idxs, order=1, mode='reflect').reshape(seg_shp)
    if PROFILE and DEBUG_LEVEL > 2:
        info_log(logger=logger, message=f'Elastic transform took {get_runtime(seconds=time.time() - t_strt)}')

    return seg


def augment(image, segmentation, rotation: bool = True, affine: bool = True, erosion: bool = True, dilation: bool = True, opening: bool = True, closing: bool = True, elastic: bool = True, logger: logging.Logger = None):
    t_strt = time.time()
    img, seg = image, segmentation

    # I. Image + segmentation
    #  1) Random rotation (whole of the image)
    if rotation:
        img, seg = random_rotation(
            image=image,
            segmentation=segmentation,
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
