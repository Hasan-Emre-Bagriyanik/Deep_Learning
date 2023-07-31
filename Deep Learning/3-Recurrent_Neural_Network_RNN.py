# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:52:25 2023

@author: Hasan Emre
"""
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%  importing the training set

data_train = pd.read_csv("Stock_Price_Train.csv")
data_train.head()

#%%
train = data_train.loc[:, ["Open"]].values
print(train)

#%% Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))   # train degerlerini normalize ediyoruz 0 ile 1 araliginda

train_scaled = scaler.fit_transform(train)
print(train_scaled)
 
#%%  # create a data structure with 50 timesteps and 1 output

x_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 1258):
    x_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#%% Reshaping

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train)
print(y_train)

#%% Create RNN Model

# Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout

# initialisinig the RNN
regressor = Sequential()

# adding the first RNN layer and some Dropout regularisation 
regressor.add(SimpleRNN(units = 100, activation="relu", return_sequences=True ,input_shape = (x_train.shape[1], 1), dropout=0.2))

# adding the second RNN layer and some Dropout regularisation 
regressor.add(SimpleRNN(units = 100, activation="relu", return_sequences=True, dropout= 0.2))

# adding the third RNN layer and some Dropout regularisation 
regressor.add(SimpleRNN(units = 100 , activation="relu", return_sequences=True,dropout=0.2))

# adding the fourth RNN layer and some Dropout regularisation 
regressor.add(SimpleRNN(units = 100))

# Adding thw output Layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer= "adam", loss = "mean_squared_error")

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 50, batch_size = 32)

#%% Getting the real stock price of 2017

data_test = pd.read_csv("Stock_Price_Test.csv")
data_test.head()

#%%

real_stock_price = data_test.loc[:, ["Open"]].values
print(real_stock_price)

#%% Getting the predicted stock price of 2017
data_total = pd.concat((data_train["Open"], data_test["Open"]), axis = 0)
inputs = data_total[len(data_total) - len(data_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs) # min max scaler
inputs

#%% 

x_test = []

for i in range(timesteps, 70):
    x_test.append(inputs[i-timesteps:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicte_stock_price = regressor.predict(x_test)
predicte_stock_price = scaler.inverse_transform(predicte_stock_price)

# visualising the results
plt.plot(real_stock_price, color = "red", label = "Real Google Stock Price")
plt.plot(predicte_stock_price, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()

print(predicte_stock_price)


