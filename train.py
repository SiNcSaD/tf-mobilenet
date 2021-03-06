import warnings

import tensorflow as tf
from tensorflow.keras import layers


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              classes=4):

    rows = input_shape[0]
    cols = input_shape[1]

    if rows != cols or rows not in [128, 160, 192, 224]:
        rows = 224
        warnings.warn('`input_shape` is undefined or non-square, '
                      'or `rows` is not in [128, 160, 192, 224]. '
                      'Weights for input shape (224, 224) will be'
                      ' loaded as the default.')

    img_input = layers.Input(shape=input_shape)

    x = _conv_block(img_input, 4, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    # x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=4)
    # x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=7)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=13)

    shape = (1, 1, int(128 * alpha))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape(shape, name='reshape_1')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1),
                      padding='same',
                      name='conv_preds')(x)
    x = layers.Reshape((classes,), name='reshape_2')(x)
    x = layers.Activation('softmax', name='act_softmax')(x)

    model = tf.keras.Model(
        img_input, x, name='mobilenet_%0.2f_%s' % (alpha, rows))
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)),
                             name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (
                                   1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


model = MobileNet(input_shape=(128, 128, 3))

model.summary()
