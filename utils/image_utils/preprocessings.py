import cv2
from configs.general_configs import (
    APPLY_CLAHE_FILTER,
    STANDARDIZE_IMAGE,
    EPSILON
)

from utils.image_utils.filters import (
    clahe_filter
)

from utils.image_utils.image_aux import (
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
    # - Convert the BGR image into a gray scale
    img = convert_to_grayscale(image)

    # - Standardization the image
    if STANDARDIZE_IMAGE:
        img = standardize(img)

    # - Apply the CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve the contrast
    # returns a 2D image (HxW)
    if APPLY_CLAHE_FILTER:
        img = clahe_filter(img)

    # - Turn image from 0-255 to 0.-1. float representation
    img = float_representation(img)

    return img
