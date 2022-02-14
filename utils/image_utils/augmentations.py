import numpy as np
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
    EROSION_SIZES,
    DILATION_SIZES,
    OPENING_SIZES,
    CLOSING_SIZES,
    SCALE_RANGE,
    SHEER_RANGE,
    CROP_SIZE,
    NON_EMPTY_CROPS,
    NON_EMPTY_CROP_THRESHOLD,
    MAX_EMPTY_CROPS,
)
from utils.image_utils.image_aux import (
    get_crop
)


def random_rotation(image: np.ndarray, segmentation: np.ndarray) -> (np.ndarray, np.ndarray):
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

    return rot_img, rot_seg


def random_crop(image, segmentation):
    h, w = CROP_SIZE, CROP_SIZE
    x = np.random.randint(0, image.shape[0] - h)

    # - y coordinate
    y = np.random.randint(0, image.shape[1] - w)

    # 2) Randomly crop the image and the label
    img_crp = get_crop(image=image, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))
    seg_crp = get_crop(image=segmentation, x=x, y=y, crop_shape=(CROP_SIZE, CROP_SIZE))

    if NON_EMPTY_CROPS and seg_crp.sum() < NON_EMPTY_CROP_THRESHOLD:
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

            print(f'The crops\' sum is {seg_crp.sum()} < {NON_EMPTY_CROP_THRESHOLD}. Trying to acquire another crop (try #{try_idx})...')

    return img_crp, seg_crp


def random_erosion(image, kernel=None):
    # Shrinks the labels
    krnl = kernel
    if krnl is None:
        krnl = np.random.choice(EROSION_SIZES)
    return grey_erosion(image, size=krnl)


def random_dilation(image, kernel=None):
    # "Fattens" the cell label
    krnl = kernel
    if krnl is None:
        krnl = np.random.choice(DILATION_SIZES)
    return grey_dilation(image, size=krnl)


def random_opening(image):
    # Connected labels are brought apart
    krnl = np.random.choice(OPENING_SIZES)
    return random_dilation(random_erosion(image, kernel=krnl), kernel=krnl)


def random_closing(image):
    # Disconnected labels are brought together
    krnl = np.random.choice(CLOSING_SIZES)
    return random_erosion(random_dilation(image, kernel=krnl), kernel=krnl)


def morphological_transform(segmentation):

    seg = segmentation
    if np.random.random() > .5:
        seg = random_erosion(seg)
    if np.random.random() > .5:
        seg = random_dilation(seg)
    if np.random.random() > .5:
        seg = random_opening(seg)
    if np.random.random() > .5:
        seg = random_closing(seg)

    # rnd_var = np.random.random()
    # if rnd_var > .5:
    #     if rnd_var > .75:
    #         seg = random_erosion(seg)
    #     else:
    #         seg = random_dilation(seg)
    # else:
    #     if rnd_var > .25:
    #         seg = random_opening(seg)
    #     else:
    #         seg = random_closing(seg)
    return seg


def affine_transform(segmentation):
    scl = np.random.uniform(*SCALE_RANGE, 2)
    tform = AffineTransform(scale=scl + 1, shear=np.random.uniform(*SHEER_RANGE))
    return warp(segmentation, tform.inverse, output_shape=segmentation.shape)


def elastic_transform(segmentation):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    seg_shp = segmentation.shape
    rnd_st = np.random.RandomState(None)

    sgm = np.random.uniform(1, 8)
    alph = np.random.uniform(50, 100)

    dx = gaussian_filter(rnd_st.rand(*seg_shp) * 2 - 1, sgm, mode='constant', cval=0) * alph
    dy = gaussian_filter(rnd_st.rand(*seg_shp) * 2 - 1, sgm, mode='constant', cval=0) * alph

    x, y = np.meshgrid(np.arange(seg_shp[0]), np.arange(seg_shp[1]))
    idxs = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(segmentation, idxs, order=1, mode='reflect').reshape(seg_shp)


def augment(image, segmentation):
    # I. Image + segmentation
    #  1) Random rotation (whole of the image)
    img, seg = random_rotation(image=image, segmentation=segmentation)

    #  2) Random crop
    img, seg = random_crop(image=img, segmentation=seg)

    # II. Segmentation only
    spoiled_seg = seg
    #  1) Non-ridged (Affine)
    if np.random.random() >= .5:
        spoiled_seg = affine_transform(seg)

    #  2) Morphological
    if np.random.random() >= .5:
        spoiled_seg = morphological_transform(spoiled_seg)

    #  3) Elastic
    if np.random.random() >= .5:
        spoiled_seg = elastic_transform(spoiled_seg)

    # img = tf.cast(img, tf.float32)
    # seg = tf.cast(seg, tf.float32)
    # spoiled_seg = tf.cast(spoiled_seg, tf.float32)

    return img, seg, spoiled_seg
