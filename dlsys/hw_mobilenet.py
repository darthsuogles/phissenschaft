import tensorflow as tf

L = tf.keras.layers
Seq = tf.keras.Sequential


def is_channels_first():
    return tf.keras.backend.image_data_format() == 'channels_first'


def _conv_block(filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    """
    channel_axis = 1 if is_channels_first() else -1
    filters = int(filters * alpha)
    return Seq([
        L.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad'),
        L.Conv2D(
            filters,
            kernel,
            padding='valid',
            use_bias=False,
            strides=strides,
            name='conv1'),
        L.BatchNormalization(axis=channel_axis, name='conv1_bn'),
        L.ReLU(6., name='conv1_relu'),
    ])


def _depthwise_conv_block(pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1,
                          name=None):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    """
    channel_axis = 1 if is_channels_first() else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    depthwise_padding = 'same' if strides == (1, 1) else 'valid'

    layers = []
    if strides != (1, 1):
        layers.append(
            L.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id))
    layers += [
        L.DepthwiseConv2D((3, 3),
                          padding=depthwise_padding,
                          depth_multiplier=depth_multiplier,
                          strides=strides,
                          use_bias=False,
                          name='conv_dw_%d' % block_id),
        L.BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id),
        L.ReLU(6., name='conv_dw_%d_relu' % block_id),
        L.Conv2D(
            pointwise_conv_filters, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_%d' % block_id),
        L.BatchNormalization(
            axis=channel_axis, name='conv_pw_%d_bn' % block_id),
        L.ReLU(6., name='conv_pw_%d_relu' % block_id),
    ]
    return Seq(layers, name=name)


def _inverted_res_block(in_channels, expansion, stride, alpha, filters,
                        block_id):
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}'.format(block_id)

    layers = []

    if block_id:
        # Expand
        layers += [
            L.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + 'expand'),
            L.BatchNormalization(
                epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN'),
            L.ReLU(6., name=prefix + 'expand_relu'),
        ]
    else:
        prefix = 'expanded_conv'

    if stride == 2:
        layers += [
            L.ZeroPadding2D(
                padding=correct_pad(backend, x, 3), name=prefix + 'pad'),
        ]

    depthwise_padding = 'same' if stride == 1 else 'valid'

    layers += [
        # Depthwise
        L.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=False,
            padding=depthwise_padding,
            name='{}_depthwise'.format(prefix)),
        L.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name='{}_depthwise_BN'.format(prefix)),
        L.ReLU(6., name='{}_depthwise_relu'.format(prefix)),

        # Project
        L.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name='{}_project'.format(prefix)),
        L.BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='{}_project_BN'.format(prefix)),
    ]

    if in_channels == pointwise_filters and stride == 1:
        layers += [L.Add(name=prefix + 'add')([inputs, x])]

    net = Seq(layers)
    return net


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    input_tensor = L.Input(shape=(3, 224, 224))

    alpha = 1.0
    depth_multiplier = 1

    depthwise_block_id = 0

    def depthwise_conv2d(num_channels, strides=(1, 1)):
        global depthwise_block_id
        depthwise_block_id += 1
        _name_scope = "depthwise_block_{}".format(depthwise_block_id)
        return _depthwise_conv_block(
            num_channels,
            alpha,
            depth_multiplier,
            strides=strides,
            block_id=depthwise_block_id,
            name=_name_scope)

    net = Seq([
        _conv_block(32, alpha, strides=(2, 2)),
        depthwise_conv2d(64),
        depthwise_conv2d(128, (2, 2)),
        depthwise_conv2d(128),
        depthwise_conv2d(256, strides=(2, 2)),
        depthwise_conv2d(256),
        depthwise_conv2d(512, strides=(2, 2)),
        depthwise_conv2d(512),
        depthwise_conv2d(512),
        depthwise_conv2d(512),
        depthwise_conv2d(512),
        depthwise_conv2d(512),
        depthwise_conv2d(1024, strides=(2, 2)),
        depthwise_conv2d(1024),
    ],
              name="backbone")

    output_tensor = net(input_tensor)

    model = tf.keras.Model(input_tensor, output_tensor)

    model.summary()
