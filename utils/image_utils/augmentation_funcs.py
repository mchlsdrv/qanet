import tensorflow as tf
from configs.general_configs import (
    BRIGHTNESS_DELTA,
    CONTRAST
)
from scipy.ndimage import grey_dilation, affine_transform, grey_erosion, grey_closing, grey_opening

def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape
    img_shp = (w, h)

    # - Represents the point around which the image will be rotated
    cX, cY = w // 2, h // 2
    rot_pt = (cX, cY)

    # - Configures the rotation matrix, which is multiplied by the image to create the rotation
    # > The first argument represents the point around which the rotation happens
    # > The second argument represents the degrees by which to rotate the image
    # > The last argument represents the scaling factor of the output image
    M = cv2.getRotationMatrix2D(rot_pt, degrees, 1.)

    # - Performs the actual rotation
    rot_img = cv2.warpAffine(image, M, img_shp)

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
