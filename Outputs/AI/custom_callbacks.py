import tensorflow as tf

def RMSE_step_t_humidity(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0, :, :, 0] - y_pred[:, 0, :, :, 0])))

def RMSE_step_t_1_humidity(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1, :, :, 0] - y_pred[:, 1, :, :, 0])))

def RMSE_step_t_wind_dir(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0, :, :, 1] - y_pred[:, 0, :, :, 1])))

def RMSE_step_t_1_wind_dir(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1, :, :, 1] - y_pred[:, 1, :, :, 1])))

def RMSE_step_t_wind_speed(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0, :, :, 2] - y_pred[:, 0, :, :, 2])))

def RMSE_step_t_1_wind_speed(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1, :, :, 2] - y_pred[:, 1, :, :, 2])))

