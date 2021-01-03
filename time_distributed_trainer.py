# # cnn lstm model

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import csv
import random
import cv2
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py
import sys


tf.get_logger().setLevel('WARNING')
print(tf.__version__)


resize_x = 100
resize_y = 100
frame_count = 64

videonames = []
video_di = {}
cats = ["deadlift", "ohp", "squat"]


dim = (frame_count,resize_x,resize_y,3)
inputShape = (dim)
In = Input(shape=inputShape, name='input_vid')
x = TimeDistributed(Conv2D(filters=50, kernel_size=(8,8), padding='same', activation='relu'))(In)
x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
x = TimeDistributed(SpatialDropout2D(0.2))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(x)
out = Dense(len(cats),activation='softmax')(x)
model = Model(inputs=In, outputs=[out])
opt = Adam(lr=1e-5, decay=1e-3 / 200)
model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])



h5f = h5py.File('data.h5', 'r')
data_size = h5f['X'].shape[0]
print(data_size)
batch_size = 1

y = utils.to_categorical(np.genfromtxt('label_list.csv', delimiter=','))

X = h5f['X'][:]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
y = y[indices]
X = X[indices]

print(sys.getsizeof(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model.fit(X_train, y_train, epochs = 4, validation_data=(X_test, y_test), shuffle=False)
model.save('model/timedistributed_model',save_format="h5")
