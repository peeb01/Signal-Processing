import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import pandas as pd


df = pd.read_csv('Dublin IRE AQI.csv')

X = df[['AQI']].values
Y = X[1:]
X = X[:-1]

past_time_data = X

future_time_data = Y


def preprocess_data(data):
    return data.reshape((data.shape[0], 1, data.shape[1]))

past_data = preprocess_data(past_time_data)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, past_data.shape[2])),
    Dense(100, activation='relu'),
    Dense(past_data.shape[2])
])


model.compile(optimizer='adam', loss='mse')

model.fit(past_data, future_time_data, epochs=10, batch_size=32)
# model.save('LSTM_AQI.h5')

future_time_point = model.predict(past_data[-1].reshape((1, 1, past_data.shape[2])))

print(future_time_point[0])

