import cv2
from configs.general_configs import (
    # APPLY_CLAHE_FILTER,
    STANDARDIZE_IMAGE,
    EPSILON,
    DEBUG_LEVEL,
)

from utils.image_funcs import (
    add_channels_dim
)


def float_representation(image):
    # - Turn image from 0-255 to 0.-1. float representation
    return image / image.max()


def normalize(image):
    # - Standardization the image
    return (image - image.min()) / (image.max() + EPSILON)


def standardize(image):
    # - Standardization the image
    return (image - image.mean()) / (image.std() + EPSILON)


def convert_to_grayscale(image):
    img = image
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = add_channels_dim(img)
    return img


def preprocess_image(image):
    if DEBUG_LEVEL > 2:
        print(f'> Preprocessing image...')
    # - Convert the BGR image into a gray scale
    if DEBUG_LEVEL > 2:
        print(f'> Converting to grayscale...')
    img = convert_to_grayscale(image)

    # - Standardization the image
    if DEBUG_LEVEL > 2:
        print(f'> Standardizing...')
    if STANDARDIZE_IMAGE:
        img = standardize(img)

    # - Turn image from 0-255 to 0.-1. float representation
    if DEBUG_LEVEL > 2:
        print(f'> Converting to float...')
    img = float_representation(img)

    if DEBUG_LEVEL > 2:
        print(f'> Done Preprocessing image...')
    return img
