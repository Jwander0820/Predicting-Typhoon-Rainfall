import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import *

from nets.mobilenetv2 import get_mobilenet_encoder
from nets.resnet50 import get_resnet50_encoder

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def resize_images(args):
    x = args[0]
    y = args[1]
    return tf.image.resize_images(x, (K.int_shape(y)[1], K.int_shape(y)[2]), align_corners=True)

def pool_block(feats, pool_factor, out_channel):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]
    #-----------------------------------------------------#
    #   分區域進行平均池化
    #   strides     = [30,30], [15,15], [10,10], [5, 5]
    #   poolsize    = 30/1=30  30/2=15  30/3=10  30/6=5
    #-----------------------------------------------------#
    pool_size = strides = [int(np.round(float(h)/pool_factor)),int(np.round(float(w)/pool_factor))]
    x = AveragePooling2D(pool_size , data_format=IMAGE_ORDERING , strides=strides, padding='same')(feats)

    #-----------------------------------------------------#
    #   利用1x1卷積進行通道數調整
    #-----------------------------------------------------#
    x = Conv2D(out_channel//4, (1 ,1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #-----------------------------------------------------#
    #   利用resize擴大特徵層面積
    #-----------------------------------------------------#
    x = Lambda(resize_images)([x, feats])
    return x

def pspnet(inputs_size,n_classes, downsample_factor=8, backbone='resnet50', aux_branch=False):
    if backbone == "mobilenet":
        #----------------------------------#
        #   獲得兩個特徵層
        #   f4為輔助分支    [30,30,96]
        #   o為主幹部分     [30,30,320]
        #----------------------------------#
        img_input, f4, o = get_mobilenet_encoder(inputs_size, downsample_factor=downsample_factor)
        out_channel = 320
    elif backbone == "resnet50":
        img_input, f4, o = get_resnet50_encoder(inputs_size, downsample_factor=downsample_factor)
        out_channel = 2048
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    #--------------------------------------------------------------#
    #	PSP模塊，分區域進行池化
    #   分別分割成1x1的區域，2x2的區域，3x3的區域，6x6的區域
    #--------------------------------------------------------------#
    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p, out_channel)
        pool_outs.append(pooled)

    #--------------------------------------------------------------------------------#
    #   利用獲取到的特徵層進行堆疊
    #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
    #--------------------------------------------------------------------------------#
    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    # 30, 30, 640 -> 30, 30, 80
    o = Conv2D(out_channel//4, (3,3), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    # 防止過擬合
    o = Dropout(0.1)(o)

    #---------------------------------------------------#
    #	利用特徵層獲得預測結果
    #   30, 30, 80 -> 30, 30, 21 -> 473, 473, 21
    #---------------------------------------------------#
    o = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(o)
    o = Lambda(resize_images)([o, img_input])

    #---------------------------------------------------#
    #   獲得每一個像素點屬於每一個類的機率
    #---------------------------------------------------#
    o = Activation("softmax", name="main")(o)

    if aux_branch:
        # 30, 30, 96 -> 30, 30, 40 
        f4 = Conv2D(out_channel//8, (3,3), data_format=IMAGE_ORDERING, padding='same', use_bias=False, name="branch_conv1")(f4)
        f4 = BatchNormalization(name="branch_batchnor1")(f4)
        f4 = Activation('relu', name="branch_relu1")(f4)
        f4 = Dropout(0.1)(f4)
        #---------------------------------------------------#
        #	利用特徵獲得預測結果
        #   30, 30, 40 -> 30, 30, 21 -> 473, 473, 21
        #---------------------------------------------------#
        f4 = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same', name="branch_conv2")(f4)
        f4 = Lambda(resize_images, name="branch_resize")([f4, img_input])

        f4 = Activation("softmax", name="aux")(f4)
        model = Model(img_input,[f4,o])
        return model
    else:
        model = Model(img_input,[o])
        return model
