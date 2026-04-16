"""Compatibility exports for historical loss function imports."""

from typhoon_rainfall.training.losses import (
    build_loss,
    cross_entropy_loss as CE,
    dice_loss_with_ce as dice_loss_with_CE,
    multi_category_focal_loss as multi_category_focal_loss1,
)


def focal_loss(gamma=1.0, alpha=0.25):
    """Legacy focal loss entrypoint kept for old notebooks."""
    return multi_category_focal_loss2(gamma=gamma, alpha=alpha)


def multi_category_focal_loss2(gamma=2.0, alpha=0.25):
    """Alternative focal loss variant kept for experiment compatibility."""
    import tensorflow as tf

    epsilon = 1e-7
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def _focal(y_true, y_pred):
        y_true_local = tf.cast(y_true, tf.float32)
        y_pred_local = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        alpha_t = y_true_local * alpha_tensor + (tf.ones_like(y_true_local) - y_true_local) * (1 - alpha_tensor)
        y_t = tf.multiply(y_true_local, y_pred_local) + tf.multiply(1 - y_true_local, 1 - y_pred_local)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1.0, y_t), gamma)
        focal = tf.multiply(tf.multiply(weight, ce), alpha_t)
        return tf.reduce_mean(focal)

    _focal.__name__ = "multi_category_focal_loss2"
    return _focal
