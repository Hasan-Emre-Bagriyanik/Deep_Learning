# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:10:25 2023

@author: Hasan Emre
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# read train
train = pd.read_csv("train.csv")
# read test
test = pd.read_csv("test.csv")

#%%
# shape
print("Train shape: ",train.shape)
print("Test shape: ",test.shape)

#%%
train.head(10)

#%%
test.head(10)

#%%
# drop label column
x_train = train.drop("label", axis = 1)
# put labels into y_train variable 
y_train = train["label"]

#%%
# value count number of digits classes
y_train.value_counts()

#%%
# plot some samples
img = np.asmatrix(x_train.iloc[0].values).reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

#%%
# plot some samples
img = np.asmatrix(x_train.iloc[7].values).reshape((28,28))
plt.imshow(img, cmap = "gray")
plt.title(train.iloc[7,0])
plt.axis("off")
plt.show()

#%%
# Normalize the data 
x_train = x_train / 255.0
test = test / 255.0
print("x_train shape: ", x_train.shape)
print("test shape: ", test.shape)

#%%
# Reshape
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ", x_train.shape)
print("test shape: ", test.shape)

#%%
# Label Encoding 
from keras.utils.np_utils import to_categorical # convertto one-hot-encoding 
y_train = to_categorical(y_train, num_classes=10)

#%%
# split the train and the validation set for the fitting 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

print("x_train shape: ", x_train.shape)
print("x_val shape: ", x_val.shape)
print("y_train shape: ", y_train.shape)
print("y_val shape: ", y_val.shape)

#%%   Convolutional Neural Network
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#%%
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "Same", activation="relu", input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
#%%
# Define the optimizer
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

#%%
model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

#%%
# Epochs and Batch Size
epochs = 40    # for better result increase the epochs
batch_size = 64


#%%
# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,   # set each sample mean to 0
    featurewise_std_normalization=False,   # divide inputs by std of the dataset
    samplewise_std_normalization= False,   # divide each input by its std
    zca_whitening=False,   # dimesion reduction
    rotation_range=5,    # randomly rotate images in the range 5 degrees
    zoom_range=0.1,      # Randomly zoom image 10%
    width_shift_range=0.1,   # randomly shift images horizontally 10%
    height_shift_range=0.1,   # randomly shift images vertically 10%
    horizontal_flip=False,     # randomly flip images
    vertical_flip=False    # randomly flip images
)  

datagen.fit(x_train)


#%%
# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)


#%%
# plot the loss and accuracy curves for training and validation 
plt.plot(history.history["val_loss"], color = "b", label = "validation Loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


#%%  Yarisma icin dosya olusturma 

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = "Label")

#%%
submission = pd.concat([pd.Series(range(1,28001), name ="ImageId"), results], axis = 1)

#%%

submission.to_csv("cnn_digit_submission.csv", index  =False)

#%%
submission.head()


