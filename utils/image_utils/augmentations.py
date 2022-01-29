import os
import pathlib
import numpy as np
import cv2
from importlib import reload
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

os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
# os.chdir('D:/University/PhD/QANET/qanet')
from configs.general_configs import (
    EROSION_SIZES,
    DILATION_SIZES,
    OPENNING_SIZES,
    CLOSING_SIZES,
    SCALE_RANGE,
    SHEER_RANGE,
    CROP_SHAPE,
    NON_EMPTY_CROPS,
    NON_EMPTY_CROP_THRESHOLD,
    MAX_EMPTY_CROPS,
)
from utils.image_utils import filters
from utils.image_utils.preprocessings import (
    preprocess_image
)
from utils.image_utils.image_aux import (
    get_seg_measure
)
from utils.visualisation_utils.plotting_funcs import (
    plot
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
    n_tries = 0
    # 1) Produce random x, y coordinates with size of crop_shape
    while True:
        # - x coordinate
        h = CROP_SHAPE[0]
        x1 = np.random.randint(0, image.shape[0] - h)
        x2 = x1 + h

        # - y coordinate
        w = CROP_SHAPE[1]
        y1 = np.random.randint(0, image.shape[1] - w)
        y2 = y1 + w

        # 2) Randomly crop the image and the label
        img_crp = image[x1:x2, y1:y2]
        seg_crp = segmentation[x1:x2, y1:y2]

        # 3) Check if one of the following happens:
        # - a - the crop contains some foreground
        # - b - if we reach the maximum number of tries
        # - c - if we don't care (i.e., non_empty=False),
        if seg_crp.sum() > NON_EMPTY_CROP_THRESHOLD \
        or n_tries >= MAX_EMPTY_CROPS \
        or not NON_EMPTY_CROPS:
            break

        n_tries += 1

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
    #  1) Non-ridged (Affine)
    spoiled_seg = affine_transform(seg)

    #  2) Morphological
    spoiled_seg = morphological_transform(spoiled_seg)

    #  3) Elastic
    if np.random.random() >= .5:
        spoiled_seg = elastic_transform(spoiled_seg)

    # img = tf.cast(img, tf.float32)
    # seg = tf.cast(seg, tf.float32)
    # spoiled_seg = tf.cast(spoiled_seg, tf.float32)

    return img, seg, spoiled_seg


DATA_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/Silver_GT/Fluo-N2DH-GOWT1-ST')
# DATA_DIR = pathlib.Path('D:/University/PhD/QANET/Data/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
IMAGE_DIR.is_dir()
GT_DIR = DATA_DIR / '01_ST/SEG'
GT_DIR.is_dir()
OUTPUT_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/output/augmentations')
# OUTPUT_DIR = pathlib.Path('D:/University/PhD/QANET/Data/output/augmentations')

if __name__ == '__main__':
    org_img = cv2.imread(f'{IMAGE_DIR}/t000.tif')
    org_seg = cv2.imread(f'{GT_DIR}/man_seg000.tif', -1)
    plot([org_img, org_img], ['Image', 'Segmentation'])

    # reload(filters)
    prep_img = preprocess_image(org_img)
    img, seg, spoiled_seg = augment(prep_img, org_seg)
    plot([img, seg, spoiled_seg], ['image', 'Segmentation', 'Augmented Segmentation'])
    img.max()
    plot([org_img, org_seg], ['Image', 'Segmentation'])

    # ROTATION
    rot_img, rot_seg = random_rotation(org_img, org_seg)
    plot([org_img, rot_img, rot_seg], ['Image', 'Rotated Image', 'Rotated Segmentation'])

    # EROSION
    plot([org_seg, random_erosion(org_seg)], ['Segmentation', 'Eroded Segmentation'])

    # DILATION
    plot([org_seg, random_dilation(org_seg)], ['Segmentation', 'Dilated Segmentation'])

    # OPENING
    plot([org_seg, random_opening(org_seg)], ['Segmentation', 'Opened Segmentation'])

    # CLOSING
    plot([org_seg, random_closing(org_seg)], ['Segmentation', 'Closed Segmentation'])

    # AFFINE TRANSFORM
    plot([org_seg, affine_transform(org_seg)], ['Segmentation', 'Affine Segmentation'])

    # ELASTIC TRANSFORM
    plot([org_seg, elastic_transform(org_seg)], ['Segmentation', 'Augmented Segmentation'])

    rot_dir = OUTPUT_DIR / 'rotations'

    er_dir = OUTPUT_DIR / 'erosions'

    dil_dir = OUTPUT_DIR / 'dilations'

    op_dir = OUTPUT_DIR / 'openings'

    cls_dir = OUTPUT_DIR / 'closiongs'

    aff_dir = OUTPUT_DIR / 'affines'

    morph_dir = OUTPUT_DIR / 'morphological'

    crops_dir = OUTPUT_DIR / 'crops'

    # TODO: This should be done together with the image loading
    for idx in range(50):

        plot([*random_rotation(org_img, org_seg)], ['Image', 'Segmentation'], rot_dir / 'rot_{idx}.png')

        plot([org_seg, random_erosion(org_seg)], ['Segmentation', 'Augmented Segmentation'], er_dir / 'er_{idx}.png')

        plot([org_seg, random_dilation(org_seg)], ['Segmentation', 'Augmented Segmentation'], dil_dir / 'dil_{idx}.png')

        plot([org_seg, random_opening(org_seg)], ['Segmentation', 'Augmented Segmentation'], op_dir / 'op_{idx}.png')

        plot([org_seg, random_closing(org_seg)], ['Segmentation', 'Augmented Segmentation'], cls_dir / 'cls_{idx}.png')

        plot([org_seg, affine_transform(org_seg)], ['Segmentation', 'Augmented Segmentation'], aff_dir / 'aff_{idx}.png')

        plot([org_seg, elastic_transform(org_seg)], ['Segmentation', 'Augmented Segmentation'], morph_dir / 'morph_{idx}.png')

    for idx in range(1000):
        img, seg, aug_seg = augment(image=org_img, segmentation=org_seg)
        J = get_seg_measure(seg, aug_seg)

        plot([img, seg, aug_seg], ['Image Crop', 'Segmentation Crop', f'Augmented Segmentation Crop (J = {J:.2f})'], save_file=crops_dir / f'clahe/aug_{idx}.png')
