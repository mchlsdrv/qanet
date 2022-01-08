import tensorflow as tf
from tensorflow import keras

from configs.general_configs import (
    EPSILON,
)


cosine_similarity_loss = keras.losses.CosineSimilarity(
    axis=-1,
    name='cosine_similarity'
)


class SCANLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, phi_X, phi_ngbrs):
        consistency_loss = entropy_loss = 0.0
        dot_prod = tf.reduce_sum(tf.multiply(phi_X, phi_ngbrs), axis=1)
        consistency_loss = tf.reduce_mean(tf.math.log(dot_prod + EPSILON))

        # IV) Calculate the entropy loss
        mean_class_probs = tf.reduce_mean(phi_X, axis=0)
        entropy = mean_class_probs * tf.math.log(mean_class_probs + EPSILON)
        entropy_loss = tf.reduce_sum(entropy)
        entropy_loss = tf.cast(entropy_loss, consistency_loss.dtype)

        loss = -consistency_loss + entropy_loss

        return loss
