import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling2D, Flatten, GRU, LSTM, RepeatVector, Dense, Input, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from build_dataset import read_inputs, read_outputs, read_test_inputs, read_test_real_outputs
from custom_callbacks import custom_loss, RMSE_step_t, RMSE_step_t_1, RMSE_both_steps

def test_results_report(real, predicted, time_spent):
    total = RMSE_both_steps(real, predicted)
    t = RMSE_step_t(real, predicted)
    t1 = RMSE_step_t_1(real, predicted)

    print(f"RMSE total: {total}")
    print(f"RMSE T: {t}")
    print(f"RMSE T+1:  {t1}")
    print(f"Tiempo total de predicción: {time_spent}")

def create_GRU_model():
    K.clear_session()
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

inputs = read_inputs()
outputs = read_outputs()

model = create_GRU_model()
model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=[RMSE_step_t, RMSE_step_t_1])

history = model.fit(inputs, outputs, epochs=100, validation_split=0.2, verbose=1, batch_size=3)

# Trained model predictions

test_inputs = read_test_inputs()
test_outputs = read_test_real_outputs()

start = time.time()
predicted_outputs = model(test_inputs)
end = time.time()

test_results_report(test_outputs, predicted_outputs, end-start)