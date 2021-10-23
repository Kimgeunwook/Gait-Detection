import sys
import os
import tensorflow as tf
import keras
from cv2 import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# import PIL.Image as pilimg
from PIL import Image
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(7)

config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.8

tf.keras.backend.set_session(tf.Session(config=config))

np.set_printoptions(threshold=np.nan)
x_train = []
y_train = []
x_test = []
y_test = []

path = './200305_gei'
list = os.listdir(path)
i = 0


with open('gei.txt', 'rb') as fr:
    x_train = pickle.load(fr)
    y_train = pickle.load(fr)
print('pickle successfully read')



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.2)

input_shape = (128, 96, 1)

batch_size = 128
num_classes = 128
epochs = 100

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss : ", score[0])
print("Test Accuracy : ", score[1])
