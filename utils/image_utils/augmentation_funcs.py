import os
import tensorflow as tf
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

from scipy.ndimage import (
    generate_binary_structure,
    grey_dilation,
    grey_erosion
)
from skimage.feature import (
    corner_harris,
    corner_subpix,
    corner_peaks
)
from skimage.transform import (
    warp,
    AffineTransform
)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
# os.chdir('D:/University/PhD/QANET/qanet')
from configs.general_configs import (
    BRIGHTNESS_DELTA,
    CONTRAST
)
CROP_SHAPE = (256, 256)
EROSION_SIZES = (5, 7, 11, 15)
DILATION_SIZES = (5, 7, 11, 15)
OPENNING_SIZES = (5, 7, 11, 15)
CLOSING_SIZES = (5, 7, 11, 15)

AFFINE_SCALE = (1.3, 1.1)
AFFINE_SHEER = .2


def random_rotation(image: np.ndarray, segmentation: np.ndarray) -> (np.ndarray, np.ndarray):
    dgrs = np.random.randint(-180, 180)
    # Rotates the image by degrees
    img_shp = image.shape
    h, w = img.shape[0], img.shape[1]

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


def random_crop(image, segmentation, crop_shape):
    # 1) Produce random x, y coordinates with size of crop_shape

    # - x coordinate
    h = crop_shape[0]
    x1 = np.random.randint(0, image.shape[0] - h)
    x2 = x1 + h

    # - y coordinate
    w = crop_shape[1]
    y1 = np.random.randint(0, image.shape[1] - w)
    y2 = y1 + w

    # 2) Randomly crop the image and the label
    img = image[x1:x2, y1:y2]
    seg = segmentation[x1:x2, y1:y2]

    return img, seg


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
    # Connected labels are brought appart
    krnl = np.random.choice(OPENNING_SIZES)
    return random_dilation(random_erosion(image, kernel=krnl), kernel=krnl)


def random_closing(image):
    # Disconnected labels are brought together
    krnl = np.random.choice(CLOSING_SIZES)
    return random_erosion(random_dilation(image, kernel=krnl), kernel=krnl)


def morphological_transform(segmentation):
    seg = segmentation
    if np.random.random() >= .5:
        seg = random_erosion(seg)
    if np.random.random() >= .5:
        seg = random_dilation(seg)
    if np.random.random() >= .5:
        seg = random_opening(seg)
    if np.random.random() >= .5:
        seg = random_closing(seg)
    return seg


def affine_transform(segmentation):
    scl = np.random.uniform(0.01, 0.1, 2)
    tform = AffineTransform(scale=scl + 1, shear=np.random.uniform(0.1, 0.2))
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

    return map_coordinates(seg, idxs, order=1, mode='reflect').reshape(seg_shp)


def augment(image, segmentation):
    # I. Image + segmentation
    #  1) Random rotation (whole of the image)
    img, seg = random_rotation(image=image, segmentation=segmentation)

    #  2) Random crop
    img, seg = random_crop(image=img, segmentation=seg, crop_shape=CROP_SHAPE)

    # II. Segmentation only
    spoiled_seg = seg
    #  1) Non-ridgid (Affine)
    if np.random.random() >= .5:
        spoiled_seg = affine_transform(seg)

    if np.random.random() >= .5:
        spoiled_seg = morphological_transform(seg)

    # img = tf.cast(img, tf.float32)
    # seg = tf.cast(seg, tf.float32)
    # spoiled_seg = tf.cast(spoiled_seg, tf.float32)

    return img, seg, spoiled_seg


DATA_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/Silver_GT/Fluo-N2DH-GOWT1-ST')
OUTPUT_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/output')
# DATA_DIR = pathlib.Path('D:/University/PhD/QANET/Data/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
IMAGE_DIR.is_dir()
GT_DIR = DATA_DIR / '01_ST/SEG'
GT_DIR.is_dir()


def plot(images):
    fig, ax = plt.subplots(1, len(images), figsize=(15, 10))
    for idx, img in enumerate(images):
        ax[idx].imshow(img, cmap='gray')
    return fig



if __name__ == '__main__':
    org_img = cv2.imread(f'{IMAGE_DIR}/t000.tif')
    org_seg = cv2.imread(f'{GT_DIR}/man_seg000.tif', -1)

    plot([org_img, org_seg])

    # ROTATION
    rot_img, rot_seg = random_rotation(img, seg)
    plot([img, rot_img, rot_seg])

    # EROSION
    er_seg = erode(seg)
    plot([seg, er_seg])

    # DILATION
    di_seg = dilate(seg)
    plot([seg, di_seg])

    # OPENING
    op_seg = open(seg)
    plot([seg, op_seg])

    # CLOSING
    cl_seg = close(seg)
    plot([seg, cl_seg])

    # AFFINE TRANSFORM
    aff_seg = affine_transform(seg)
    plot([seg, aff_seg])

    # ELASTIC TRANSFORM
    el_seg = elastic_transform(seg)
    plot([seg, el_seg])

    rot_dir = OUTPUT_DIR / 'rotations'
    er_dir = OUTPUT_DIR / 'erosions'
    dil_dir = OUTPUT_DIR / 'dilations'
    op_dir = OUTPUT_DIR / 'openings'
    cls_dir = OUTPUT_DIR / 'closiongs'
    aff_dir = OUTPUT_DIR / 'affines'
    morph_dir = OUTPUT_DIR / 'morphological'

    img, seg, spoiled_seg = augment(org_img, org_seg)
    plot([img, seg, spoiled_seg])
    for idx in range(10):

        rot = plot([img, *random_rotation(img, seg)])
        rot.savefig(f'{rot_dir}/rot_{idx}.png')
        plt.close(rot)

        er = plot([seg, random_erosion(seg)])
        er.savefig(f'{er_dir}/er_{idx}.png')
        plt.close(er)

        dil = plot([seg, random_dilation(seg)])
        dil.savefig(f'{dil_dir}/dil_{idx}.png')
        plt.close(dil)

        op = plot([seg, random_opening(seg)])
        op.savefig(f'{op_dir}/op_{idx}.png')
        plt.close(op)

        cls = plot([seg, random_closing(seg)])
        cls.savefig(f'{cls_dir}/cls_{idx}.png')
        plt.close(cls)

        aff = plot([seg, affine_transform(seg)])
        aff.savefig(f'{aff_dir}/aff_{idx}.png')
        plt.close(aff)

        morph = plot([seg, elastic_transform(seg)])
        morph.savefig(f'{morph_dir}/morph_{idx}.png')
        plt.close(morph)
