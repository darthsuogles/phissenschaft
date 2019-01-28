"""Identity Mappings in Deep Residual Networks
"""
import tensorflow as tf

import numpy as np

from collections import namedtuple
from dlsys.pixsrl import show_graph

L = tf.keras.layers
Seq = tf.keras.Sequential
print(tf.keras.__version__)


def conv2d(out_channels, kernel_size, strides=1):
    return L.Conv2D(
        filters=out_channels,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        data_format="channels_first")


def conv2d_bn(out_channels, kernel_size, strides=1):
    return Seq([
        conv2d(out_channels, kernel_size, strides),
        L.BatchNormalization(),
        L.ReLU(),
    ])


def linear_cls(num_classes):
    return Seq([
        L.GlobalAveragePooling2D(data_format="channels_first"),
        L.Dense(4, activation="relu"),
        L.Softmax(),
    ])


def build_model():
    return Seq([
        conv2d_bn(32, kernel_size=7, strides=3),
        conv2d_bn(64, kernel_size=3),
        linear_cls(4),
    ])

try: print(iss)
except: iss = tf.InteractiveSession()

# sess.run(tf.global_variables_initializer())
# ss = sess.run(
#     probs, feed_dict={input_tensor: np.random.randn(1, 3, 80, 80)})

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    model = build_model()
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    train_data = np.random.randn(32, 3, 80, 80)
    train_labels = np.random.randint(4, size=(32, 1))
    model.fit(train_data, train_labels, batch_size=4, epochs=10)

show_graph(graph)

class Block(namedtuple("Block", ["scope", "unit_fn", "args"])):
    """A named tuple describing a ResNet block.
      Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
          returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
          contains one (depth, depth_bottleneck, stride) tuple for each unit in the
          block to serve as argument to unit_fn.
      """


class BlockSpec(
        namedtuple("BlockSpec", ["depth", "depth_bottleneck", "stride"])):
    """ A list of length equal to the number of units in the `Block`.
        The list contains one (depth, depth_bottleneck, stride) tuple for
        each unit in the block to serve as argument to unit_fn.
    """


def conv2d_same(inputs: tf.Tensor,
                num_outputs: int,
                kernel_size,
                stride,
                rate=1,
                data_format='channels_first',
                scope=None):
    if 1 == stride:
        conv_impl = L.Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation_rate=rate,
            padding='same',
            data_format=data_format)
    else:
        effective_kernel_size = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = effective_kernel_size - 1
        pad_begin = pad_total // 2
        pad_end = pad_total - pad_begin
        """
        | existing | pad |
        |oooooooooo|ooooo|
        """
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
        conv_impl = L.Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation_rate=rate,
            padding='valid',
            data_format=data_format)

    return conv_impl(inputs)


def stack_blocks_dense(inputs: tf.Tensor,
                       blocks: list,
                       output_stride=None,
                       store_non_stride_activations=False,
                       outputs_collections=None):
    curr_stride = 1
    dilation_rate = 1
    features = inputs

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [inputs]) as scope:
            for i, unit in enumerate(block.args):
                if store_non_stride_activations and i == len(block.args):
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope(
                        "unit_{}".format(i + 1), values=[features]):
                    if output_stride is not None and curr_stride == output_stride:
                        features = block.unit_fn(
                            features,
                            rate=dilation_rate,
                            **dict(unit, stride=1))
                        dilation_rate *= unit.get('stride', 1)

                    else:
                        features = block.unit_fn(features, rate=dilation_rate)
                        curr_stride *= unit.get('stride', 1)
                        if output_stride is not None and curr_stride > output_stride:
                            raise ValueError(
                                "the target `output_stride` cannot be reached")

            # TODO: collect activations
            # TODO: subsample block output activations

    return features
