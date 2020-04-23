import sys
import os
import tensorflow as tf
import keras
# import cv2
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

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

# (x_train, y_train)
#
# (x_test, y_test)

# im = pilimg.open('./GEI/__001_bg-01_000.png')
# im.show()
np.set_printoptions(threshold=np.nan)
# pix = np.array(im)
#
# print(pix)
# path = './GEI/__001_bg-02_054.png'
# # img = Image.open(path).convert('L')
# img = imread(path,cv2.IMREAD_GRAYSCALE)
# img_numpy = np.array(img, 'uint8')
# imshow("img", img)
# print(img_numpy)
# waitKey(0)
# destroyAllWindows()

x_train = []
y_train = []
x_test = []
y_test = []

path = './200305_gei'
list = os.listdir(path)
i = 0


# for file in list:
#     imgpath = path + '/' + file
#     img = imread(imgpath, cv2.IMREAD_GRAYSCALE)
#     pix = np.array(img, 'uint8')
#     x_train.append(pix)
#     sss = file.split('_')
#     label = sss[2]
#     y_train.append(int(label))

# for file in list:
#     imgpath = path + '/' + file
#     angle = int(file[-7:-4])
#     if angle == 0 or angle == 54 or angle == 90:
#         img = imread(imgpath, cv2.IMREAD_GRAYSCALE)
#         pix = np.array(img, 'uint8')
#         x_train.append(pix)
#         print(file + ' appended to x_train')
#         sss = file.split('_')
#         label = sss[2]
#         y_train.append(int(label))
#     elif angle > 125:
#         img = imread(imgpath, cv2.IMREAD_GRAYSCALE)
#         pix = np.array(img, 'uint8')
#         x_test.append(pix)
#         print(file + ' appended to x_test')
#         sss = file.split('_')
#         label = sss[2]
#         y_test.append(int(label))



#피클쓰기
# with open('gei.txt', 'wb') as f:
#     pickle.dump(x_train, f)
#     pickle.dump(y_train, f)
#
# print(x_train)
# print(y_train)



#피클읽기
with open('gei.txt', 'rb') as fr:
    x_train = pickle.load(fr)
    y_train = pickle.load(fr)
print('pickle successfully read')
#
# for file in list:
#     imgpath = path + '/' + file
#     img = imread(imgpath, cv2.IMREAD_GRAYSCALE)
#     pix = np.array(img, 'uint8')
#     x_train.append(pix)
#     x_test.append(pix)
#     # sss = file.split('_')
#     # label = sss[2]
#     if file[0:8] == 'geunwook':
#         y_train.append(125)
#         y_test.append(125)
#     elif file[0:6] == 'sunkyu':
#         y_train.append(126)
#         y_test.append(126)
#     elif file[0:4] == 'jiho':
#         y_train.append(127)
#         y_test.append(127)

#테스트용
# with open('gei.txt', 'rb') as fr:
#     xxx = pickle.load(fr)
#     yyy = pickle.load(fr)
# print('pickle successfully read')




#######################################################################################################################

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.2)

input_shape = (128, 96, 1)

batch_size = 128
num_classes = 128
epochs = 100

x_train = np.array(x_train)
x_test = np.array(x_test)
#x_test = np.array(xxx)
y_train = np.array(y_train)
y_test = np.array(y_test)
#y_test = np.array(yyy)


x_train = np.expand_dims(x_train, axis=3)
#y_train = np.expand_dims(y_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
#y_test = np.expand_dims(y_test, axis=3)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print(y_train)

# print('x_train : ', np.array(x_train).shape)
# print('x_test : ', np.array(x_test).shape)
# print('y_train : ', np.array(y_train).shape)
# print('y_test : ', np.array(y_test).shape)
#
# print('x_train : ', x_train[0])
# print('x_test : ', x_test[0])
# print('y_train : ', y_train[0])
# print('y_test : ', y_test[0])

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
