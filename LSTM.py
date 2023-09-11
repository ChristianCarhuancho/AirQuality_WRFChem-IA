import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling2D, Flatten, LSTM, RepeatVector, Dense, Input


def create_model():
    model = Sequential()

    model.add(Input(shape=(3, 4, 100)))
    model.add(TimeDistributed(Conv1D(50, 2)))
    model.add(TimeDistributed(Conv1D(25, 2)))
    model.add(TimeDistributed(Flatten()))

    # Encoder
    model.add(LSTM(100, activation='relu'))

    model.add(RepeatVector(2))

    # Decoder
    model.add(LSTM(100, activation='relu', return_sequences=True))

    # Outputs
    model.add(TimeDistributed(Dense(30)))

    return model

model = create_model()
model.compile(optimizer='adam', loss='mse')
print(model.summary())


#wandb.init(entity='wandb', project='LSTM-AIR_QUALITY')
#history = model.fit(X, Y, epochs=1000, validation_split=0.2, verbose=1, batch_size=3, callbacks=[WandbCallback()])