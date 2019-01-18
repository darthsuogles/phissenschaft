import tensorflow as tf
import numpy as np

layers = tf.keras.layers
K = tf.keras.backend
slim = tf.contrib.slim

from tools.hack import *


def randn(*dims):
    return np.random.randn(*dims).astype(np.float32)


def test_tf_layers():
    sess = tf.InteractiveSession()

    input_tensor = tf.placeholder(
        tf.float32, [None, 16, 16, 3], name='input_tensor')
    normalized = tf.layers.batch_normalization(input_tensor)

    sess.run(tf.global_variables_initializer())
    outs = sess.run([normalized, input_tensor],
                    {input_tensor: randn(16, 16, 16, 3)})


def test_keras_layers():
    input_shape = (4, 4, 7)
    model = tf.keras.models.Sequential()
    norm_layer = layers.BatchNormalization(
        axis=-1, input_shape=input_shape, momentum=0.8, fused=True)
    model.add(norm_layer)
    model.compile(
        loss='mse', optimizer=tf.train.GradientDescentOptimizer(0.01))

    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, *input_shape))
    model.fit(x, x, epochs=4, verbose=1)
    out = model.predict(x)
    # both `beta` and `gamma` have a shape of last input dimension,
    # as is specified in the model construction. In image processing domain,
    # this can be interpreted as normalizing channel-wise
    out -= np.reshape(K.eval(norm_layer.beta), (1, 1, 1, input_shape[-1]))
    out /= np.reshape(K.eval(norm_layer.gamma), (1, 1, 1, input_shape[-1]))

    np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)
