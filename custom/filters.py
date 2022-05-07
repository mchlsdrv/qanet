import numpy as np
import cv2
from configs.general_configs import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)
# from utils.image_utils.image_aux import (
from utils.image_funcs import (
    add_channels_dim
)


def clahe_filter(image: np.ndarray) -> np.ndarray:
    img = image
    # - Convert the image to grayscale if:
    # -a- the image has channel dimension
    # -b- the number of channels is 3
    if len(image.shape) > 2 \
            and image.shape[-1] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    img = clahe.apply(img.astype('uint8'))

    # - Add the channels dimension
    img = add_channels_dim(img)

    return img
