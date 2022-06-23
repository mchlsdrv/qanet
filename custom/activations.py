import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.utils import tf_utils


class Swish(Layer):
    """Continuous activation function
  It allows a small gradient when the unit is not active:
  ```
    f(x) = x*sigmoid(x)
  ```
  Usage:
  >>> layer = tf.keras.layers.Swish()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.9, -0.3, 0.0, 2.0]
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as the input.
  """

    def __init__(self):
        super().__init__()
        self.supports_masking = True

    def call(self, inputs):
        return tf.keras.activations.swish(inputs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
