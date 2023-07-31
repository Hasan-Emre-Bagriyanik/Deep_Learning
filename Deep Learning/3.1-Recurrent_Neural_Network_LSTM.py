# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:00:37 2023

@author: Hasan Emre
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
data = pd.read_csv("international-airline-passengers.csv")
data.head()

#%% 

data = data.iloc[:,1].values
plt.plot(data)
plt.xlabel("Time")
plt.ylabel("Number of Passenger")
plt.title("International airline passenger")
plt.show()

#%% preprocessing data

#%%
#reshape
data = data.reshape(-1,1)
data = data.astype("float32")
data.shape

#%% scaling

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

#%% train test split

train_size = int(len(data) * 0.50)
test_size = len(data) - train_size
train = data[0:train_size,:]
test = data[train_size:len(data), :]
print("Train size: {}, test size: {} ".format(len(train), len(test)))

#%% timesteps

timesteps = 10
datax = []
datay = []

for i in range(len(train) - timesteps -1):
    a = train[i:(i+timesteps), 0]
    datax.append(a)
    datay.append(train[i + timesteps, 0])
trainx = np.array(datax)
trainy = np.array(datay)

#%%

datax = []
datay = []

for i in range(len(test) - timesteps - 1):
    a = test[i:(i + timesteps), 0]
    datax.append(a)
    datay.append(test[i + timesteps, 0])
testx = np.array(datax)
testy = np.array(datay)

#%% 

trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
testx = np.reshape(testx, (testx.shape[0], 1 , testx.shape[1]))

#%% create LSTM model

model = Sequential()
model.add(LSTM(20, input_shape= (1, timesteps)))  # 20 LSTM neuron (block)

model.add(Dense(units=1))
model.compile(loss= "mean_squared_error", optimizer = "adam")
model.fit(trainx, trainy, epochs=100, batch_size = 1)

#%% predictions and visualising LSTM model

train_pred = model.predict(trainx)
test_prep = model.predict(testx)

# invert predictions
train_pred = scaler.inverse_transform(train_pred)
trainy = scaler.inverse_transform([trainy])
test_prep = scaler.inverse_transform(test_prep)
testy = scaler.inverse_transform([testy])

# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(trainy[0], train_pred[:,0]))
print("Train Score: %.2f RMSE" %(train_score))

test_score = math.sqrt(mean_squared_error(testy[0], test_prep[:,0]))
print("Test Scroe: %.2f RMSE" %(test_score))

#%%  visualising LSTM model

# shifting train
train_pred_plot = np.empty_like(data)
train_pred_plot[:,:] = np.nan
train_pred_plot[timesteps: len(train_pred) + timesteps, :] = train_pred

# shifting test predictions for plotting 
test_pred_plot = np.empty_like(data)
test_pred_plot[:,:] = np.nan
test_pred_plot[len(train_pred) + (timesteps * 2) + 1 : len(data) - 1, :]  = test_prep

# plot baseline and predictions
plt.plot(scaler.inverse_transform(data)) 
plt.plot(train_pred_plot)
plt.plot(test_pred_plot)
plt.show()
















