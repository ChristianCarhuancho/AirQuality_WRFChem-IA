from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, TimeDistributed, MaxPooling2D, Flatten, GRU, RepeatVector, Dense, Input, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam

from build_dataset import read_inputs, read_outputs
from custom_callbacks import RMSE_step_t_humidity, RMSE_step_t_1_humidity, RMSE_step_t_wind_dir, RMSE_step_t_1_wind_dir, RMSE_step_t_wind_speed, RMSE_step_t_1_wind_speed


def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def create_model():
    model = Sequential()

    model.add(Input(shape=(3, 7, 4, 14)))
    model.add(TimeDistributed(Conv2D(50, (2,2), activation='relu', kernel_initializer='glorot_normal', kernel_regularizer='l2')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU()))
    model.add(TimeDistributed(Flatten()))

    # Encoder
    model.add(GRU(100, activation='relu', recurrent_activation='relu', kernel_regularizer='l2', dropout=0.1, recurrent_dropout=0.1))

    model.add(RepeatVector(2))

    # Decoder
    model.add(GRU(100, activation='relu', recurrent_activation='relu', kernel_regularizer='l2', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

    # Outputs
    model.add(TimeDistributed(Dense(units=49, kernel_initializer='glorot_normal', kernel_regularizer='l2', activation="relu")))

    model.add(TimeDistributed(Dense(units=98, kernel_initializer='glorot_normal', kernel_regularizer='l2', activation="relu")))

    model.add(TimeDistributed(Reshape((14,7,1))))

    model.add(TimeDistributed(Conv2DTranspose(10, (5,5), strides=(5,5), padding='same', kernel_initializer='glorot_normal', kernel_regularizer='l2')))

    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(LeakyReLU()))

    model.add(TimeDistributed(Conv2DTranspose(3, (5,5), strides=(1,1), padding='same', kernel_initializer='glorot_normal', kernel_regularizer='l2')))

    return model


model = create_model()
#model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=[RMSE_step_t, RMSE_step_t_1])

model.compile(optimizer=Adam(clipvalue=0.5, learning_rate=1e-6), loss='mean_squared_error', metrics=[RMSE_step_t_humidity, RMSE_step_t_1_humidity, RMSE_step_t_wind_dir, RMSE_step_t_1_wind_dir, RMSE_step_t_wind_speed, RMSE_step_t_1_wind_speed])
print(model.summary())

inputs = read_inputs()
scaler = StandardScaler()

batch_size, time_steps, X, Y, nFeatures = inputs.shape
inputs_reshaped = inputs.reshape(batch_size * time_steps * X * Y, nFeatures)

# Inicializar el scaler y aplicar la normalización
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs_reshaped)

# Revertir el cambio de forma
inputs_scaled = inputs_scaled.reshape(batch_size, time_steps, X, Y, nFeatures)
inputs = inputs_scaled

outputs = read_outputs()

print('Inputs nan: ', np.sum(np.isnan(inputs)))
print('Outputs nan: ', np.sum(np.isnan(outputs)))

wandb.init(entity='carhuanchochristian', project='GRU-ATMOS')
history = model.fit(inputs, outputs, epochs=100, validation_split=0.2, verbose=1, batch_size=10, callbacks=[WandbCallback()])

#history = model.fit(inputs, outputs, epochs=100, validation_split=0.2, verbose=1, batch_size=10)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()
#plt.savefig('foo.png')

