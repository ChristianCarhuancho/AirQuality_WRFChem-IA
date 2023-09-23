import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling2D, Flatten, LSTM, RepeatVector, Dense, Input

from build_dataset import read_inputs, read_outputs

def create_model():
    tf.keras.backend.clear_session()
    model = Sequential()

    model.add(Input(shape=(3, 4, 99)))
    model.add(TimeDistributed(Conv1D(50, 2)))
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

inputs = read_inputs()
outputs = read_outputs()


wandb.init(entity='carhuanchochristian', project='LSTM-AIR_QUALITY')
history = model.fit(inputs, outputs, epochs=100, validation_split=0.2, verbose=1, batch_size=3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('foo.png')
