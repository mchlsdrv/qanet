import os
import io
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
import cv2
from utils.image_utils import preprocessing_funcs
import matplotlib.pyplot as plt

STANDARDIZE_CROPS = False
EPSILON = 1e-7,
APPLY_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BAR_HEIGHT = 70


def preprocessings(image):
    # - Convert the BGR image into a gray scale
    img = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)

    # - Turn image from 0-255 to 0.-1. float representation
    img = img / img.max()

    # - Standirdization the image
    if STANDARDIZE_CROPS:
        img = (img - img.mean()) / (img.std() + EPSILON)

    # - Allpy the CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve the contrast
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        img = clahe.apply(img)

    return img


def load_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) < 3:
        img = np.expand_dims(img, 2)
    return img


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
