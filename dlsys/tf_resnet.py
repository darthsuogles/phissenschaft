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

To use resnet in some application:

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


num_classes = 17

def resnet_tiny_fn(features, labels, mode):
    block_gen = resnet_v2.resnet_v2_block
    blocks = [
        block_gen('block1', base_depth=1, num_units=2, stride=2),
        block_gen('block2', base_depth=2, num_units=2, stride=2),
        block_gen('block3', base_depth=4, num_units=2, stride=2),
        block_gen('block4', base_depth=8, num_units=2, stride=1),
    ]
    nominal_stride = 8  # 2 * 2 * 2 * 1

    is_training = tf.estimator.ModeKeys.TRAIN == mode

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        with slim.arg_scope([slim.layers.batch_norm], is_training=is_training):
            logits, endpoints = resnet_v2.resnet_v2(
                features['x'], blocks,
                num_classes=num_classes,
                is_training=is_training,
                # without `global_pool`, output must match block reduction
                global_pool=True,
                output_stride=None,
                include_root_block=True,
                reuse=tf.AUTO_REUSE,
                scope='resnet_tiny')

    # This only requires the indices, rather than one-hot labels
    losses = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=losses,
        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=losses, train_op=train_op)


# We do NOT have to provide as session for this
train_size = 100
train_features = np.random.randn(train_size, 114, 114, 16).astype(np.float32)
train_labels = np.random.randint(low=0, high=num_classes, size=train_size).astype(np.int32)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_features},
    y=train_labels,
    batch_size=4,
    num_epochs=None,
    shuffle=True)

estimator = tf.estimator.Estimator(
    model_fn=resnet_tiny_fn,
    model_dir="/tmp/resnet_tiny_dir")

estimator.train(input_fn=train_input_fn,
                steps=100)

# For a bridge between slim.arg_scope and tf.keras.layers
# https://github.com/tensorflow/models/blob/master/research/object_detection/builders/hyperparams_builder.py
