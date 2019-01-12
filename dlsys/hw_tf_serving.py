import tensorflow as tf
from pathlib import Path
from dlsys.pixsrl import show_graph

from tensorflow.python.keras.layers.convolutional import Conv as ConvType

L = tf.keras.layers

def conv_block_impl(tag: str,
                    features: tf.Tensor,
                    layer_impl: ConvType,
                    num_out_channles: int,
                    kernel_size: tuple,
                    strides: tuple,
                    padding: str,
                    dilation_rate: tuple,
                    use_bias: bool):
    net = features
    activation_first = layer_impl is L.SeparableConv2D
    if activation_first:
        net = L.Activation('relu', name="{}_act".format(tag))(net)

    net = layer_impl(num_out_channles,
                     kernel_size=kernel_size,
                     strides=strides,
                     dilation_rate=dilation_rate,
                     padding=padding,
                     use_bias=use_bias,
                     name="{}".format(tag))(net)
    net = L.BatchNormalization(name="{}_bn".format(tag))(net)

    if not activation_first:
        net = L.Activation('relu', name="{}_act".format(tag))(net)

    return net


def conv_block(tag: str,
               features: tf.Tensor,
               layer_impl: ConvType,
               num_out_channles: int,
               kernel_size: tuple,
               strides=(1, 1),
               padding="same",
               dilation_rate=(1, 1),
               use_bias=False):
    kwargs = locals()
    return conv_block_impl(**kwargs)


def residual_block(features: tf.Tensor, num_out_channels: int):
    residual = L.Conv2D(num_out_channels, (1, 1), strides=(2, 2),
                        padding="same", use_bias=False)(features)
    residual = L.BatchNormalization()(residual)
    return residual


def entry_flow_block(tag: str, features: tf.Tensor, num_out_channles: int):
    with tf.name_scope(tag):
        residual = residual_block(features, num_out_channles)

        with tf.name_scope("sepconv"):
            features = conv_block("sepconv1", features,
                                  L.SeparableConv2D, num_out_channles, (3, 3))
            features = conv_block("sepconv2", features,
                                  L.SeparableConv2D, num_out_channles, (3, 3))

            features = L.MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                                      name="exit_pool")(features)

        features = L.add([features, residual])
        return features

def middle_flow_block(tag: str, features: tf.Tensor):
    with tf.name_scope(tag):
        residual = features

        with tf.name_scope("sepconv"):
            for idx in range(3):
                features = conv_block("sepconv{}".format(idx), features,
                                      L.SeparableConv2D, 728, (3, 3))

        features = L.add([residual, features])
        return features

def exit_flow_block(features: tf.Tensor):
    residual = residual_block(features, 1024)

    with tf.name_scope("sepconv"):
        features = conv_block("sepconv1", features,
                              L.SeparableConv2D, 728, (3, 3))
        features = conv_block("sepconv2", features,
                              L.SeparableConv2D, 1024, (3, 3))
        features = L.MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                                  name="exit_pool")(features)

    features = L.add([features, residual])

    features = tf.keras.Sequential([
        L.SeparableConv2D(1536, (3, 3)),
        L.Activation("relu"),
        L.SeparableConv2D(2048, (3, 3)),
        L.Activation("relu"),
        L.GlobalAveragePooling2D(),
    ], name="tail")(features)

    return features


def model(image_tensor: tf.Tensor):
    # Initial blocks
    with tf.name_scope("init_conv"):
        feats = conv_block("block1_conv1", image_tensor,
                           L.Conv2D, 32, (3, 3), strides=(2, 2))
        feats = conv_block("block1_conv2", feats,
                           L.Conv2D, 64, (3, 3))

    with tf.name_scope("entry_flow"):
        # TODO: w.r.t. paper, the first block of entry flow does not have
        #       a leading ReLU
        for idx, entry_num_channels in enumerate([128, 256, 728]):
            feats = entry_flow_block("block{}".format(idx), feats,
                                     entry_num_channels)

    with tf.name_scope("middle_flow"):
        for idx in range(8):
            feats = middle_flow_block("block{}".format(idx), feats)

    with tf.name_scope("exit_flow"):
        feats = exit_flow_block(feats)

    # Show current process (not part of the graph)
    show_graph(tf.get_default_graph())

if __debug__:

    try: print(sess)
    except: sess = tf.InteractiveSession()

    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(0)

    image_tensor = tf.placeholder(
        dtype=tf.float32, shape=[None, 299, 299, 3], name="image_tensor")

    model(image_tensor)
