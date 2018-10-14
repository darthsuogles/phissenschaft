"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""

from tools.hack import *
import tensorflow as tf
slim = tf.contrib.slim
import_tensorflow_models()
_THIS_MODULE = git_repo_root() / 'dlsys'

import numpy as np

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.core import freezable_batch_norm
from object_detection.protos import hyperparams_pb2

from slim.nets import resnet_utils, resnet_v2

sess = tf.InteractiveSession()
# Clean the graph every once in a while
# tf.reset_default_graph()

def get_random_tensor(name: str, random_values: np.ndarray):
    return tf.convert_to_tensor(
        random_values.astype(np.float32),
        dtype=tf.float32,
        name=name)

def get_randn_tensor(name: str, d1, *args):
    shape = [d1] + list(args)
    return get_random_tensor(name, np.random.randn(*shape))


features = get_randn_tensor('input_tensor', 1, 224, 224, 3)
subsampled_features = slim.max_pool2d(
    features, [1, 1], stride=2, scope=None)

resnet_graph = tf.Graph()
with resnet_graph.as_default():
    features = tf.placeholder(
        dtype=tf.float32, shape=[None, 513, 513, 3],
        name='input_tensor')
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_101(features,
                                                  21,
                                                  is_training=False,
                                                  global_pool=False,
                                                  output_stride=16)

layers = tf.keras.layers

resnet_tiny_graph = tf.Graph()
with resnet_tiny_graph.as_default():
    block_gen = resnet_v2.resnet_v2_block
    blocks = [
        block_gen('block1', base_depth=1, num_units=2, stride=2),
        block_gen('block2', base_depth=2, num_units=2, stride=2),
        block_gen('block3', base_depth=4, num_units=2, stride=2),
        block_gen('block4', base_depth=8, num_units=2, stride=1),
    ]
    nominal_stride = 8  # 2 * 2 * 2 * 1

    input_features = tf.placeholder(
        dtype=tf.float32, shape=[1, 114, 114, 16])

    logits, endpoints = resnet_v2.resnet_v2(input_features, blocks,
                                            num_classes=17,
                                            is_training=False,
                                            # without `global_pool`, output must match block reduction
                                            global_pool=True,
                                            output_stride=None,
                                            include_root_block=True,
                                            scope='resnet_tiny')

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        with slim.arg_scope([slim.layers.batch_norm], is_training=False):
            output_features = resnet_utils.stack_blocks_dense(
                input_features, blocks, nominal_stride)
