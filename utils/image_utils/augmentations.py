import tensorflow as tf
from configs.general_configs import (
    BRIGHTNESS_DELTA,
    CONTRAST
)


def image_augmentations(image):
    img = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)  # Equivalent to adjust_brightness() using a delta randomly picked in the interval [-max_delta, max_delta)
    img = tf.image.random_contrast(img, lower=CONTRAST[0], upper=CONTRAST[1])  # Equivalent to adjust_contrast() but uses a contrast_factor randomly picked in the interval [lower, upper).
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img
