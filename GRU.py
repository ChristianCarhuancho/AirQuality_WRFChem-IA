import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling2D, Flatten, GRU, RepeatVector, Dense, Input, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam

from build_dataset import read_inputs, read_outputs
from custom_callbacks import custom_loss, RMSE_step_t, RMSE_step_t_1

def create_model():
    model = Sequential()

    model.add(Input(shape=(3, 4, 99)))
    model.add(TimeDistributed(Conv1D(50, 2, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer='l2')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(LeakyReLU()))
    model.add(TimeDistributed(Flatten()))

    # Encoder
    model.add(GRU(100, activation='relu', recurrent_activation='relu', kernel_regularizer='l2', dropout=0.1, recurrent_dropout=0.1))

    model.add(RepeatVector(2))

    # Decoder
    model.add(GRU(100, activation='relu', recurrent_activation='relu', kernel_regularizer='l2', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

    # Outputs
    model.add(TimeDistributed(Dense(units=18, kernel_initializer='glorot_normal', kernel_regularizer='l2', activation="relu")))

    model.add(TimeDistributed(Dense(units=36, kernel_initializer='glorot_normal', kernel_regularizer='l2', activation="relu")))

    model.add(TimeDistributed(Reshape((6,2,3))))

    model.add(TimeDistributed(Conv2DTranspose(10, (5,5), strides=(3,3), padding='same', kernel_initializer='glorot_normal', kernel_regularizer='l2')))

    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(LeakyReLU()))

    model.add(TimeDistributed(Conv2DTranspose(1, (5,5), strides=(3,3), padding='same', kernel_initializer='glorot_normal', kernel_regularizer='l2')))

    return model


model = create_model()
model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=[RMSE_step_t, RMSE_step_t_1])
print(model.summary())

inputs = read_inputs()
outputs = read_outputs()

wandb.init(entity='carhuanchochristian', project='GRU-AIR_QUALITY')
history = model.fit(inputs, outputs, epochs=100, validation_split=0.2, verbose=1, batch_size=3, callbacks=[WandbCallback()])
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()
#plt.savefig('foo.png')
