import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,Conv3D,
                            MaxPooling2D, Concatenate, UpSampling2D,
                            Conv3DTranspose, MaxPooling3D,concatenate,
                            UpSampling3D,BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers as opt
from keras import regularizers
from keras.regularizers import l2, l1
from keras.layers.core import SpatialDropout3D
import tensorflow as tf
from keras.losses import mean_squared_error, binary_crossentropy, kullback_leibler_divergence, categorical_hinge, hinge
from keras_radam import RAdam

def switch_norm(x, scope='switch_norm') :
    with tf.variable_scope(scope) :
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], keep_dims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3, 4], keep_dims=True)
        tf.get_variable_scope().reuse_variables()
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred): #94.25% / 90.66%
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
def bce_dice_with_l2_loss(y_true, y_pred): #94.35% / 90.71%  #94.57% / 90.68%
    l2_loss = mean_squared_error(y_true, y_pred)
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + 0.01 * l2_loss # 0.1 * l2_loss / 1e-5
# D+l2
# D+0.1*l2: 93.09% / 92.59%
# D+0.01*l2: 93.68% / 92.94% v
# D+0.001*l2: 93.41% / 92.08% 

# binary_crossentropy
# 0.1: 93.86% / 92.88%
# 0.01: 93.60% / 92.61% 
# 0.001: 93.54% / 92.89% v
# 0.0001 93.40% / 92.77%
# 0.00001 93.83% / 92.66%

# dice_loss
# 0.1: 93.10% / 92.70%
# 0.01: 93.35% / 92.64%
# 0.001: 92.01% / 91.88%

# l2_loss
# 0.1: 93.19% / 92.32%
# 0.01: 93.61% / 92.83% v
# 0.001: 93.80% / 92.58% 
# 0.0001: 93.67 / 92.48%
# 0.00001: 93.30 / 92.54%

def h_dice_with_l2_loss(y_true, y_pred): # 94.04% / 91.34%
    l2_loss = mean_squared_error(y_true, y_pred)
    return hinge(y_true, y_pred) + dice_loss(y_true, y_pred) + 1e-5 * l2_loss 

def ch_dice_with_l2_loss(y_true, y_pred): # 93.70% / 91.39%
    l2_loss = mean_squared_error(y_true, y_pred)
    return categorical_hinge(y_true, y_pred) + dice_loss(y_true, y_pred) + 1e-5 * l2_loss 


def dice_with_l2_KL_loss(y_true, y_pred):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = dice_coef(y_true, y_pred)
    KL_loss = kullback_leibler_divergence(y_true, y_pred)
    total_loss = dice_loss + 0.1 * l2_loss + 0.1* KL_loss
    return total_loss

def dice_with_l2_loss(y_true, y_pred, weight_l2=1.0):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = dice_coef(y_true, y_pred)
    total_loss = weight_l2 * l2_loss + (1.0 - weight_l2)*dice_loss
    return total_loss

def crossentropy_with_l2(y_true, y_pred, weight_l2=2.0):
    crossentropy = binary_crossentropy(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)
    return crossentropy + l2 * weight_l2


def create_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(2,2),
                        deconvolution_kernel_size=(2,2),
                        pool_size=(2,2),
                        strides=(2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.001):
    """
    Create a 2D Unet model

    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model
    
def diunet3D(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model

    
def diunet3D_SReLU(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', activation = 'selu', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=bce_dice_with_l2_loss, metrics=[dice_coefficient])
    return model


def diunet3D_LeakyReLU(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(0.3)(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(0.3)(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(0.3)(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(0.3)(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(0.3)(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(0.3)(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(0.3)(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(0.3)(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(0.3)(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(0.3)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(0.3)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(0.3)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(0.3)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(0.3)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(0.3)(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=RAdam(lr=init_lr, warmup_proportion=0.1, min_lr=1e-5), loss=bce_dice_with_l2_loss, metrics=[dice_coefficient])
    return model

def diunet3D_PReLU(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5):
    filters = 28
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(inputs)
    conv1 = PReLU()(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv1)
    conv1 = PReLU()(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool1)
    conv2 = PReLU()(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv2)
    conv2 = PReLU()(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool2)
    conv3 = PReLU()(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv3)
    conv3 = PReLU()(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool3)
    conv4 = PReLU()(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = PReLU()(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 3, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = PReLU()(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = PReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = PReLU()(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv6)
    conv6 = PReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), padding='same', dilation_rate = 2, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv6)
    conv6 = PReLU()(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv7)
    conv7 = PReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), padding='same', dilation_rate = 1, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(conv7)
    conv7 = PReLU()(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=bce_dice_with_l2_loss, metrics=[dice_coefficient])
    return model

def unet(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    #conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    #conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    #conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model

def mdiunet3D(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay))(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

#    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(pool3)
#    conv4 = BatchNormalization(axis = 1)(conv4)
#    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv4)
#    conv4 = BatchNormalization(axis = 1)(conv4)
#    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv4)
#    conv4 = BatchNormalization(axis = 1)(conv4)

#    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
#    conv5 = SpatialDropout3D(0.1)(up5)
#    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv5)
#    conv5 = BatchNormalization()(conv5)
#    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv5)
#    conv5 = BatchNormalization()(conv5)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv3), conv2])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 2, kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv1])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay))(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', dilation_rate = 1, kernel_regularizer=l2(weight_decay))(conv6)

    conv7 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv6)
    model = Model(input=inputs, output=conv7)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model   

def unet3D(input_image_size, n_labels=1, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-4):
    filters = 32
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Conv3D(filters*1, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Conv3D(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Conv3D(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding='same')(conv3)

    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Conv3D(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv4), conv3])
    conv5 = SpatialDropout3D(0.1)(up5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3DTranspose(filters*8, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv5), conv2])
    conv6 = SpatialDropout3D(0.1)(up6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3DTranspose(filters*4, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=4)([UpSampling3D(size=(2, 2, 2))(conv6), conv1])
    conv7 = SpatialDropout3D(0.1)(up7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3DTranspose(filters*2, kernel_size=(3, 3, 3), activation = 'elu', padding='same', kernel_regularizer=l2(weight_decay))(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)
    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=opt.Adam(lr=init_lr), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model

def get_unet(input_image_size):
    inputs = Input(shape=input_image_size)
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

    conv51 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    conv52 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    conv61 = Conv3DTranspose(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3DTranspose(256, (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    conv71 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conc42], axis=4)
    conv81 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

#    model.summary()
    #plot_model(model, to_file='model.png')

#    model.compile(optimizer=opt.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt.Adam(lr=1e-3), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model

def res_unet(input_image_size):
    inputs = Input(shape=input_image_size)
    conv1 = Conv3D(32, (3, 3, 3), activation='elu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='elu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(64, (3, 3, 3), activation='elu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='elu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(128, (3, 3, 3), activation='elu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='elu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = Conv3D(256, (3, 3, 3), activation='elu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='elu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)

    conv5 = Conv3D(512, (3, 3, 3), activation='elu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='elu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='elu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='elu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='elu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='elu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='elu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='elu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='elu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='elu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=opt.Adam(lr=1e-3), loss=loss_dice_coefficient_error, metrics=[dice_coefficient])
    return model



def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=32,
                        convolution_kernel_size=(5,5,5),
                        deconvolution_kernel_size=(5,5,5),
                        pool_size=(2,2,2),
                        strides=(2,2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001):
    """
    Create a 3D Unet model
    Example
    -------
    unet_model = create_unet_model3D( (128,128,128,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    weight_decay = 1E-4
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])
        
        if i == 0:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            dilation_rate = 1,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(weight_decay))(inputs)
            conv = BatchNormalization()(conv)
        else:

            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            dilation_rate= 2,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(weight_decay))(pool)
            conv = BatchNormalization()(conv)
        encoding_convolution_layers.append(Conv3D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        dilation_rate= 4,
                                                        activation='relu',
                                                        padding='same', kernel_regularizer=l2(weight_decay))(conv))

        if i < len(layers)-1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])
            
    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)

        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same',dilation_rate= 4, kernel_regularizer=l2(weight_decay))(outputs)
        tmp_deconv = UpSampling3D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=4)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])
        outputs = BatchNormalization()(outputs)
        outputs = Conv3DTranspose(filters=number_of_filters, kernel_size=convolution_kernel_size, dilation_rate= 2,
                        activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Conv3DTranspose(filters=number_of_filters, kernel_size=convolution_kernel_size, dilation_rate= 1,
                        activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(outputs)
        outputs = BatchNormalization()(outputs)
    if mode == 'classification':
        if number_of_classification_labels == 1:

            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1),
                            activation='sigmoid')(outputs)

        else:

            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1),
                            activation='softmax')(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model