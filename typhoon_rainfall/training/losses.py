"""Loss functions used by the rainfall segmentation models.

These functions keep names compatible with the original notebook code where
possible, because Keras history/checkpoint monitor names depend on function
names in several places.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


def dice_loss_with_ce(beta: float = 1.0, smooth: float = 1e-5):
    """Return a combined categorical cross-entropy and Dice loss function."""
    def _dice_loss_with_ce(y_true, y_pred):
        y_pred_local = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce_loss = -y_true * K.log(y_pred_local)
        ce_loss = K.mean(K.sum(ce_loss, axis=-1))

        tp = K.sum(y_true * y_pred_local, axis=[0, 1, 2])
        fp = K.sum(y_pred_local, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true, axis=[0, 1, 2]) - tp

        score = ((1 + beta**2) * tp + smooth) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + smooth
        )
        dice_loss = 1 - tf.reduce_mean(score)
        return ce_loss + dice_loss

    _dice_loss_with_ce.__name__ = "dice_loss_with_CE"
    return _dice_loss_with_ce


def cross_entropy_loss():
    """Return the custom cross-entropy closure kept for legacy imports."""
    def _ce(y_true, y_pred):
        y_pred_local = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce_loss = -y_true * K.log(y_pred_local)
        return K.mean(K.sum(ce_loss, axis=-1))

    _ce.__name__ = "CE"
    return _ce


def multi_category_focal_loss(alpha, gamma: float = 2.0):
    """Return the multi-class focal loss used in earlier experiments."""
    epsilon = 1e-7
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def _focal(y_true, y_pred):
        y_true_local = tf.cast(y_true, tf.float32)
        y_pred_local = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_t = tf.multiply(y_true_local, y_pred_local) + tf.multiply(
            1 - y_true_local, 1 - y_pred_local
        )
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1.0, y_t), gamma)
        focal = tf.matmul(tf.multiply(weight, ce), alpha_tensor)
        return tf.reduce_mean(focal)

    _focal.__name__ = "multi_category_focal_loss1"
    return _focal


def build_loss(use_focal_loss: bool, num_classes: int):
    """Choose the configured loss for model.compile."""
    if use_focal_loss:
        alpha = [[index + 1] for index in range(num_classes)]
        return multi_category_focal_loss(alpha=alpha, gamma=2.0)
    return "categorical_crossentropy"
