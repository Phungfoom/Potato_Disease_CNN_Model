import tensorflow as tf

import config


@tf.keras.utils.register_keras_serializable(package="potato_id")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss for integer (sparse) class labels.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    where p_t is the model's probability for the true class.
    gamma=0 reduces to standard cross-entropy.
    """

    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)  # safe with mixed_float16 policy
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), y_true], axis=1)
        probs_true = tf.gather_nd(y_pred, indices)
        focal_weight = tf.pow(1.0 - probs_true, self.gamma)
        return focal_weight * (-tf.math.log(probs_true))

    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma}


def focal_loss():
    return SparseCategoricalFocalLoss(gamma=config.FOCAL_LOSS_GAMMA)
