import numpy as np
import cv2


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BAR_HEIGHT = 70


def clahe_filter(image):
    # - Remove the information bar from the image
    img = image[:-BAR_HEIGHT]

    # - If the image is in RGB - convert it to gray scale
    if img.shape[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # - Allpy the CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve the contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    img = clahe.apply(img)

    # - If the image does not have the channels dimension - expand its' dimensions
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    return img
