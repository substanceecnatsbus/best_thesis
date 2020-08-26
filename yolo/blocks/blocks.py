import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, Reshape, add, concatenate

def convolutional(x, filters, kernel_size, strides, batch_norm, block_number):

    if strides == 1:
        padding = "same"
    else:
        padding = "valid"
        x = ZeroPadding2D(((1, 0), (1, 0)), name = f"ZeroPad_{block_number}")(x)

    x = Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        use_bias = not batch_norm,
        name = f"Convolutional_{block_number}"
    )(x)

    if batch_norm:
        x = BatchNormalization(name = f"BatchNormalize_{block_number}")(x)
        x = LeakyReLU(alpha = 0.1, name = f"LeakyRelu_{block_number}")(x)

    return x

def upsample(x, size, block_number):
     x = UpSampling2D(size, name = f"Upsample_{block_number}")(x)
     return x

def shortcut(x, prev, block_number):
    x = add([x, prev], name = f"Shortcut_{block_number}")
    return x

def route(layers, block_number):
    if len(layers) > 1:
        x = concatenate(layers, axis = -1, name = f"Route_{block_number}")
    else:
        x = layers[0]
    return x

def yolo(x, num_anchors, num_classes, block_number):
    x_shape = x.get_shape()
    x = Reshape((x_shape[1], x_shape[2], num_anchors, num_classes + 5), name = f"Yolo_{block_number}")(x)
    return x