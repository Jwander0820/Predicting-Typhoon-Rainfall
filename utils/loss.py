from random import shuffle

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

def dice_loss_with_CE(beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        CE_loss = - y_true * K.log(y_pred)  
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true * y_pred, axis=[0,1,2]) #axis或許為(len(train),128,128,18)，然後取0,1,2做計算?
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true, axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()) 
        #clip 元素級裁減，裁剪最大值與最小值 K.epsilon=1e-7 

        CE_loss = - y_true * K.log(y_pred)  
        #keras中交叉損失函數定義!! (參考byhttps://blog.csdn.net/MrR1ght/article/details/93649259
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE



def focal_loss(gamma=1., alpha=.25):  #常見的版本，但用起來怪怪的
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed



def multi_category_focal_loss1(alpha, gamma=2.0):  #較佳；本研究所使用的focal loss
    """source from https://blog.csdn.net/u011583927/article/details/90716942
    https://github.com/monkeyDemon/AI-Toolbox/blob/master/computer_vision/image_classification_keras/loss_function/focal_loss.py
    focal loss for multi category of multi label problem
    適用於多分類或多標籤問題的focal loss
    alpha用於指定不同類別/標籤的權重，數組大小需要與類別個數一致
    當你的數據集不同類別/標籤之間存在偏斜，可以嘗試本函數做為loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[[1],[2],[3],][2]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    #alpha = tf.constant([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18]], dtype=tf.float32)
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed
    
    
    
def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    適用於多分類或多標籤問題的focal loss
    alpha控制真值y_true為1/0时的權重
        1的權重为alpha, 0的權重为1-alpha
    当你的模型欠擬合，學習存在困難時，可以嘗試適用本函数作為loss
    当模型過於激進(無論何時總是傾向預測出1),嘗試將alpha調小
    当模型過於惰性(無論何時總是傾向預測出0,或是某一個固定的常數,說明没有學到有效特徵)
        嘗試將alpha調大,鼓勵模型進行預測出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed
