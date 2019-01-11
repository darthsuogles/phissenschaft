"""Identity Mappings in Deep Residual Networks
"""
import tensorflow as tf

from collections import namedtuple

slim = tf.contrib.slim
layers = tf.keras.layers

from tensorflow.contrib.slim import nets
from tensorflow.contrib.slim.nets import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope


class Block(namedtuple("Block", ["scope", "unit_fn", "args"])):
    pass


def conv2d_same(inputs: tf.Tensor,
                num_outputs: int,
                kernel_size,
                stride,
                rate=1,
                data_format='channels_first',
                scope=None):
    if 1 == stride:
        conv = layers.Conv2D(
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
        conv = layers.Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation_rate=rate,
            padding='valid',
            data_format=data_format)

    return conv(inputs)


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
