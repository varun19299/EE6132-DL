'''
Model Definition for MNIST.

'''
import tensorflow as tf

def create_model(data_format):
    """Model to recognize digits in the MNIST dataset.

    Args:
    * data_format: NHWC or NCHW. Prefer latter for CPUs, former for GPUs.

    Returns:
      A tf.keras.Model.
    """
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)),
            l.Conv2D(
                32,
                3,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                32,
                3,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(500, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10)
        ])