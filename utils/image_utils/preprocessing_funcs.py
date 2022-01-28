import numpy as np
import cv2


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BAR_HEIGHT = 70
STANDARDIZE_CROPS = True

def preprocessings(image):
    # - Turn image from 0-255 to 0.-1. float representation
    img = image / image.max()

    # - Standirdization the image
    if STANDARDIZE_CROPS:
        img = (img - img.mean()) / (img.std() + EPSILON)

    # - If the image is 2D - add the channel dimension
    if len(img.shape) < 3:
        img = np.expand_dims(img, 2)
    # - If the image is in RGB - convert it to gray scale
    elif img.shape[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # - Allpy the CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve the contrast
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        img = clahe.apply(img)

    return img
