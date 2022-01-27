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
from skimage.transform import (
    warp,
    AffineTransform
)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
os.chdir('D:/University/PhD/QANET/qanet')
from configs.general_configs import (
    BRIGHTNESS_DELTA,
    CONTRAST
)
from utils.general_utils import aux_funcs

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

    return map_coordinates(segmentation, idxs, order=1, mode='reflect').reshape(seg_shp)


def augment(image, segmentation):
    # I. Image + segmentation
    #  1) Random rotation (whole of the image)
    img, seg = random_rotation(image=image, segmentation=segmentation)

    #  2) Random crop
    img, seg = random_crop(image=img, segmentation=seg, crop_shape=CROP_SHAPE)

    # II. Segmentation only
    #  1) Non-ridgid (Affine)
    spoiled_seg = affine_transform(seg)

    #  2) Morphological
    spoiled_seg = morphological_transform(spoiled_seg)

    #  3) Elastic
    if np.random.random() >= .5:
        spoiled_seg = elastic_transform(spoiled_seg)

    # img = tf.cast(img, tf.float32)
    # seg = tf.cast(seg, tf.float32)
    # spoiled_seg = tf.cast(spoiled_seg, tf.float32)

    # plot([img, seg, spoiled_seg])
    return img, seg, spoiled_seg


def plot(images, labels, save_file=None):
    fig, ax = plt.subplots(1, len(images), figsize=(15, 10))
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)


# DATA_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/Silver_GT/Fluo-N2DH-GOWT1-ST')
DATA_DIR = pathlib.Path('D:/University/PhD/QANET/Data/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
IMAGE_DIR.is_dir()
GT_DIR = DATA_DIR / '01_ST/SEG'
GT_DIR.is_dir()
# OUTPUT_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/output/augmentations')
OUTPUT_DIR = pathlib.Path('D:/University/PhD/QANET/Data')

if __name__ == '__main__':
    org_img = cv2.imread(f'{IMAGE_DIR}/t000.tif')
    org_seg = cv2.imread(f'{GT_DIR}/man_seg000.tif', -1)

    plot([org_img, org_seg], ['Image', 'Segmentation'])

    # ROTATION
    rot_img, rot_seg = random_rotation(img, seg)
    plot([img, rot_img, rot_seg], ['Image', 'Rotated Image', 'Rotated Segmentation'])

    # EROSION
    er_seg = random_erosion(seg)
    plot([seg, er_seg], ['Segmentation', 'Eroded Segmentation'])

    # DILATION
    di_seg = random_dilation(seg)
    plot([seg, di_seg], ['Segmentation', 'Dilated Segmentation'])

    # OPENING
    op_seg = random_opening(seg)
    plot([seg, op_seg], ['Segmentation', 'Opened Segmentation'])

    # CLOSING
    cl_seg = random_closing(seg)
    plot([seg, cl_seg], ['Segmentation', 'Closed Segmentation'])

    # AFFINE TRANSFORM
    aff_seg = affine_transform(seg)
    plot([seg, aff_seg], ['Segmentation', 'Affine Segmentation'])

    # ELASTIC TRANSFORM
    el_seg = elastic_transform(org_seg)
    plot([org_seg, el_seg], ['Segmentation', 'Augmented Segmentation'])

    OUTPUT_DIR.is_dir()
    augs_dir = OUTPUT_DIR / 'augmentations'
    os.makedirs(augs_dir, exist_ok=True)

    rot_dir = augs_dir / 'rotations'
    os.makedirs(rot_dir, exist_ok=True)

    er_dir = augs_dir / 'erosions'
    os.makedirs(er_dir, exist_ok=True)

    dil_dir = augs_dir / 'dilations'
    os.makedirs(dil_dir, exist_ok=True)

    op_dir = augs_dir / 'openings'
    os.makedirs(op_dir, exist_ok=True)

    cls_dir = augs_dir / 'closiongs'
    os.makedirs(cls_dir, exist_ok=True)

    aff_dir = augs_dir / 'affines'
    os.makedirs(aff_dir, exist_ok=True)

    morph_dir = augs_dir / 'morphological'
    os.makedirs(morph_dir, exist_ok=True)

    all_crops_dir = augs_dir / 'all cropped'
    os.makedirs(all_crops_dir, exist_ok=True)

    img, seg, spoiled_seg = augment(org_img, org_seg)
    plot([img, seg, spoiled_seg])

    for idx in range(50):

        rot = plot([*random_rotation(org_img, org_seg)], ['Image', 'Segmentation'])
        rot.savefig(f'{rot_dir}/rot_{idx}.png')
        plt.close(rot)

        er = plot([org_seg, random_erosion(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        er.savefig(f'{er_dir}/er_{idx}.png')
        plt.close(er)

        dil = plot([org_seg, random_dilation(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        dil.savefig(f'{dil_dir}/dil_{idx}.png')
        plt.close(dil)

        op = plot([org_seg, random_opening(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        op.savefig(f'{op_dir}/op_{idx}.png')
        plt.close(op)

        cls = plot([org_seg, random_closing(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        cls.savefig(f'{cls_dir}/cls_{idx}.png')
        plt.close(cls)

        aff = plot([org_seg, affine_transform(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        aff.savefig(f'{aff_dir}/aff_{idx}.png')
        plt.close(aff)

        morph = plot([org_seg, elastic_transform(org_seg)], ['Segmentation', 'Augmented Segmentation'])
        morph.savefig(f'{morph_dir}/morph_{idx}.png')
        plt.close(morph)

    
    for idx in range(1000):
        img, seg, aug_seg = augment(image=org_img, segmentation=org_seg)
        J = aux_funcs.get_seg_measure(seg, aug_seg)

        plot([img, seg, aug_seg], ['Image Crop', 'Segmentation Crop', f'Augmented Segmentation Crop (J = {J:.2f})'], save_file=f'{all_crops_dir}/aug_{idx}.png')
