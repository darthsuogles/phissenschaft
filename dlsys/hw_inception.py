import tensorflow as tf

L = tf.keras.layers
Seq = tf.keras.Sequential

tf.keras.backend.set_image_data_format("channels_first")

def is_channels_first():
    return tf.keras.backend.image_data_format() == 'channels_first'

def conv2d_bn(filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              dilation_rate=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 1 if is_channels_first() else 3
    return Seq([
        L.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name),
        L.BatchNormalization(axis=bn_axis, scale=False, name=bn_name),
        L.Activation('relu', name=name)
    ], name=None)


def stem():
    return Seq([
        conv2d_bn(32, 3, 3, strides=(2, 2), padding='valid'),
        conv2d_bn(32, 3, 3, padding='valid'),
        conv2d_bn(64, 3, 3),
        L.MaxPooling2D((3, 3), strides=(2, 2)),
        conv2d_bn(80, 1, 1, padding='valid'),
        conv2d_bn(192, 3, 3, padding='valid'),
        L.MaxPooling2D((3, 3), strides=(2, 2)),
    ])


def inception_module(input_shape: tuple, pool_filters: int, name: str):
    input_tensor = L.Input(shape=input_shape)

    branch1x1 = conv2d_bn(64, 1, 1)(input_tensor)

    branch5x5 = Seq([
        conv2d_bn(48, 1, 1),
        conv2d_bn(64, 5, 5),
    ], name="branch-5x5")(input_tensor)

    branch3x3dbl = Seq([
        conv2d_bn(64, 1, 1),
        conv2d_bn(96, 3, 3),
        conv2d_bn(96, 3, 3),
    ], name="branch-3x3-duo")(input_tensor)

    branch_pool = Seq([
        L.AveragePooling2D((3, 3),
                           strides=(1, 1),
                           padding='same'),
        conv2d_bn(pool_filters, 1, 1)
    ], name="branch-pool")(input_tensor)

    output_tensor = L.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name=name)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


def check_shape(input_tensor, *shape_dims):
    if is_channels_first():
        shape = tuple(input_tensor.shape[1:].as_list())
    else:
        _s = input_tensor.shape
        shape = (_s[3], _s[1], _s[2])

    assert shape == shape_dims, (shape, shape_dims)


def InceptionV3(**kwargs):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    tf.reset_default_graph()

    input_shape = (3, 299, 299)
    input_tensor = L.Input(shape=input_shape)
    channel_axis = 1 if is_channels_first() else 3

    x = stem()(input_tensor)
    check_shape(x, 192, 35, 35)

    x = inception_module(x.shape[1:], pool_filters=32,
                         name="mixed0")(x)
    check_shape(x, 256, 35, 35)
    x = inception_module(x.shape[1:], pool_filters=64,
                         name="mixed1")(x)
    check_shape(x, 288, 35, 35)
    x = inception_module(x.shape[1:], pool_filters=64,
                         name="mixed2")(x)
    check_shape(x, 288, 35, 35)

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(384, 3, 3, strides=(2, 2), padding='valid')(x)

    branch3x3dbl = Seq([
        conv2d_bn(64, 1, 1),
        conv2d_bn(96, 3, 3),
        conv2d_bn(96, 3, 3, strides=(2, 2), padding='valid'),
    ])(x)

    branch_pool = L.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = L.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')



    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    return x
