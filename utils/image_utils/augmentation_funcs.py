import tensorflow as tf
import pathlib
from tensorflow.nn import (
    erosion2d,
    dilation2d,
)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import (
    apply_affine_transform,
)
import os
os.chdir('D:/University/PhD/QANET/qanet')
from configs.general_configs import (
    BRIGHTNESS_DELTA,
    CONTRAST
)

FILTERS = (5, 5, 1)
STRIDES=(1, 1, 1, 1)
PADDING='SAME'
DATA_FORMAT='NHWC'
DILATIONS=(1, 1, 1, 1)


def erode(image):
    return erosion2d(value=image, filters=FILTERS, strides=STRIDES, padding=PADDING, data_format=DATA_FORMAT, dilations=DILATIONS)


def dilate(image):
    return dilation2d(value=image, filters=FILTERS, strides=STRIDES, padding=PADDING, data_format=DATA_FORMAT, dilations=DILATIONS)


def open(image):
    return dilation(erosion(image))


def close(image):
    return erosion(dilation(image))


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    img_shp = image.shape
    h, w = img.shape[0], img.shape[1]

    # - Represents the point around which the image will be rotated
    cX, cY = w // 2, h // 2
    rot_pt = (cX, cY)

    # - Configures the rotation matrix, which is multiplied by the image to create the rotation
    # > The first argument represents the point around which the rotation happens
    # > The second argument represents the degrees by which to rotate the image
    # > The last argument represents the scaling factor of the output image
    M = cv2.getRotationMatrix2D(rot_pt, degrees, 1.)

    # - Performs the actual rotation
    rot_img = cv2.warpAffine(image, M, img_shp[:-1])

    return rot_img


def get_random_crop(image, segmentation, crop_shape):
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

    img = tf.cast(img, tf.float32)
    seg = tf.cast(seg, tf.float32)

    return img, seg


def augmentations(image, segmentation):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img


DATA_DIR = pathlib.Path('D:/University/PhD/QANET/Data/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
IMAGE_DIR.is_dir()
GT_DIR = DATA_DIR / '01_ST/SEG'
GT_DIR.is_dir()


def plot(images):
    fig, ax = plt.subplots(1, len(images), figsize=(15, 10))
    for idx, img in enumerate(images):
        ax[idx].imshow(img, cmap='gray')


if __name__ == '__main__':
    img = cv2.imread(f'{IMAGE_DIR}/t000.tif')
    seg = cv2.imread(f'{GT_DIR}/man_seg000.tif', -1)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(seg, cmap='gray')

    plot([img, seg])

    rot_img = rotate(img, 45)
    plot([img, rot_img])
    g = np.expand_dims(seg, [0, -1])
    g.shape
    eros_seg = erode(tf.cast(np.expand_dims(seg, [0, -1]), tf.int16))
    plot([img, rot_img])
    for idx, btch in enumerate(dl):
        imgs, segs, mod_segs, J = btch
        print(J)
        plt.imshow(imgs[0], cmap='gray')
        # print(idx)
        # print(btch[0].shape)
