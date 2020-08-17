import tensorflow as tf
import numpy as np
import tensorflow.python.keras.backend as k


# Purpose: Define the IOU metric to optimize for when fitting the model
# Parameters:
#   y_true: ground truth binary mask
#   y_pred: predicted binary mask
def mean_iou(y_true, y_pred):
    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_conv = tf.cast((y_pred > t), tf.int32)
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_conv, 2)
        k.get_session().run(tf.compat.v1.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)

    return k.mean(k.stack(prec), axis=0)
