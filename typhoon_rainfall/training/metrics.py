"""Segmentation metrics used during training.

The metric names intentionally mirror the historical notebook names because
the training callback monitors `val__RMSE`. Please be careful when renaming
these closures, as it can change history CSV columns and checkpoint behavior.
"""

from __future__ import annotations

import tensorflow as tf

SMOOTH = 1.0


def Iou_score(smooth: float = SMOOTH, threhold: float = 0.5):
    """Return mean IoU-style score over all classes."""
    del threhold

    def _Iou_score(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
        score = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(score, axis=[0, 1])

    return _Iou_score


def f_score(beta: float = 1.0, smooth: float = SMOOTH, threhold: float = 0.5):
    """Return macro F-score/Dice-style metric over predicted probability maps."""
    del threhold

    def _f_score(y_true, y_pred):
        tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        fp = tf.reduce_sum(y_pred, axis=[1, 2]) - tp
        fn = tf.reduce_sum(y_true, axis=[1, 2]) - tp
        score = ((1 + beta**2) * tp + smooth) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + smooth
        )
        return tf.reduce_mean(score, axis=[0, 1])

    return _f_score


def precision(smooth: float = SMOOTH, threhold: float = 0.5):
    """Return precision metric closure."""
    del threhold

    def metric_precision(y_true, y_pred):
        tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        fp = tf.reduce_sum(y_pred, axis=[1, 2]) - tp
        score = (tp + smooth) / (tp + fp + smooth)
        return tf.reduce_mean(score, axis=[0, 1])

    return metric_precision


def recall(smooth: float = SMOOTH, threhold: float = 0.5):
    """Return recall metric closure."""
    del threhold

    def metric_recall(y_true, y_pred):
        tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        fn = tf.reduce_sum(y_true, axis=[1, 2]) - tp
        score = (tp + smooth) / (tp + fn + smooth)
        return tf.reduce_mean(score, axis=[0, 1])

    return metric_recall


def mean_RMSE(smooth: float = SMOOTH, threhold: float = 0.5):
    """Return RMSE computed from argmax label maps."""
    del smooth, threhold

    def _RMSE(y_true, y_pred):
        pred_labels = tf.argmax(y_pred, axis=-1)
        true_labels = tf.argmax(y_true, axis=-1)
        rmse = (tf.reduce_sum((pred_labels - true_labels) ** 2, axis=[1, 2]) / 4700) ** 0.5
        return tf.reduce_mean(rmse * 1.0)

    return _RMSE


def blank_RMSE(smooth: float = SMOOTH, threhold: float = 0.5):
    """Return baseline RMSE against an all-blank prediction."""
    del smooth, threhold

    def _blank(y_true, y_pred):
        del y_pred
        true_labels = tf.argmax(y_true, axis=-1)
        rmse = (tf.reduce_sum(true_labels**2, axis=[1, 2]) / 4700) ** 0.5
        return tf.reduce_mean(rmse * 1.0)

    _blank.__name__ = "_blank"
    return _blank


def default_metrics():
    """Return the standard metric list used by training."""
    from tensorflow.keras.metrics import CategoricalAccuracy

    return [
        Iou_score(),
        f_score(),
        precision(),
        recall(),
        CategoricalAccuracy(name="acc"),
        mean_RMSE(),
        blank_RMSE(),
    ]
