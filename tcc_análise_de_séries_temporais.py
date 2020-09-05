# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import seaborn as sns

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd

data = pd.read_csv('BTC.csv')

data = data[::-1].reset_index(drop=True)

data.head()

data.reset_index(inplace=True)
data.drop(columns=['Date', 'Symbol', 'Volume BTC', 'index'], inplace = True)

data.shape

data.head()

data.shape

data['Close'].plot(legend = True, figsize = (18, 6))

pip install pandas_ta

import pandas_ta as ta

aobvdf = ta.aobv(close=data['Close'], volume=data['Volume USD'], mamode='sma', fast=10, slow=20)
data['OBV'] = aobvdf['OBV']

data = data.fillna(0)

data.head()

data['Delta'] = [data.iloc[index+1]['Close'] - data.iloc[index]['Close'] if index < len(data)-1 else 0 for index, row in data.iterrows()]
data = data[:-1]

data.head()

data['Delta'].plot(kind='line')

def digitize(n):
    if n >= 0:
        return 1
    return 0
    
data['to_predict'] = data['Delta'].apply(lambda d: digitize(d))

data.head()

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :dataset.shape[1] - 1]
		dataX.append(a)
		dataY.append(dataset[i+look_back, -1])
	return np.array(dataX), np.array(dataY)

data = data.to_numpy()

look_back = 100
X, Y = create_dataset(data, look_back)

X.shape

Y.shape

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

batch_size = 16
nb_epoch = 300
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# fit network
history = model.fit(X, Y, epochs=nb_epoch, batch_size=batch_size, validation_split=0.20, shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

