import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# download data
data = yf.download("BTC-USD", start="2020-01-01")
data = data[['Close']]

# scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# create dataset
x = []
y = []

for i in range(60, len(scaled_data)):
    x.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

x, y = np.array(x), np.array(y)

# model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, epochs=5, batch_size=32)

# prediction
pred = model.predict(x)

plt.plot(pred)
plt.title("BTC Prediction")
plt.show()
