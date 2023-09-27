import tensorflow as tf

def custom_loss(y_true, y_pred):
    loss_t = tf.losses.mean_squared_error(y_true[:, 0, :], y_pred[:, 0, :])
    loss_t_1 = tf.losses.mean_squared_error(y_true[:, 1, :], y_pred[:, 1, :])
    return loss_t, loss_t_1

def RMSE_step_t(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0, :] - y_pred[:, 0, :])))

def RMSE_step_t_1(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1, :] - y_pred[:, 1, :])))