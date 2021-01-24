import os
import re
import cv2
import sys
import json
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Input, add, Conv2D, Flatten, Dense, Dropout, LSTM





sys.path.insert(1, '../..')
from src.utils.visualizer import visualize_generator
from src.utils.hyperparameters import get_hyperparam_obj, update_hyperparams
from src.generator.custom_generator import time_series_generator

hyperparams = {
    "num_epochs": 10,
    "batch_size": 1,
    "height": 224,
    "width": 224,
    "lr": .0001,
    "seq_len": 64,
    "categories": ["deadlift", "ohp", "squat"],
    "loss_opt": "SGD"
}

#save hyperparams locally to json
update_hyperparams(hyperparams)

#format hyperparams dictionary as an object
config = get_hyperparam_obj()



val_dir = '/Users/Bryce/Documents/GitHub/workout_tracker_data/val'
train_dir = '/Users/Bryce/Documents/GitHub/workout_tracker_data/train'

temp_batch_size = config.batch_size
config.batch_size = 32
gen = time_series_generator(train_dir, config)
visualize_generator(gen)
config.batch_size = temp_batch_size

steps_per_epoch = len(glob.glob(train_dir + "/*")) // config.batch_size
validation_steps = len(glob.glob(val_dir + "/*")) // config.batch_size


def get_td_cnn_lstm():
    activation = 'relu'
    input = Input(shape = (config.seq_len, config.width, config.height, 3))
    x = TimeDistributed(Conv2D(32, (3,3), padding='same', name = "con1"))(input)
    x = TimeDistributed(MaxPooling2D())(x)
    # x = TimeDistributed(Conv2D(64,(2,2), activation=activation, name = "con2"))(x)
    # x = TimeDistributed(MaxPooling2D())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(32, activation=activation, name="den3"))(x)
    x = LSTM(32, return_sequences=False, dropout=0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(.3)(x)
    main_output = Dense(len(config.categories), activation = 'softmax', name='main_output')(x)
    model = Model(inputs=input, outputs=main_output)
    return model

def get_model():
    pretrained_cnn = load_model("../../models/custom_dataset_model3_finetuned")
    for layer in pretrained_cnn.layers:
        layer.trainable = False
    input_layer = Input(shape=(config.seq_len, config.width, config.height, 3))
    curr_layer = TimeDistributed(pretrained_cnn)(input_layer)
    curr_layer = TimeDistributed(Dense(1))(curr_layer)
    lstm_out = LSTM(64, return_sequences=False, dropout=.5)(curr_layer)
    x = Dense(64, activation='relu')(lstm_out)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.3)(x)
    main_output = Dense(len(config.categories), activation = 'softmax', name='main_output')(x)
    model = Model(inputs=input_layer, outputs=main_output)
    model.summary()
    return model

if config.loss_opt == "SGD":
    opt = tf.keras.optimizers.SGD(learning_rate=config.lr, momentum=0.0, name="SGD")
else:
    opt = tf.keras.optimizers.Adam(learning_rate=config.lr,name='Adam')

callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='../../models/cnn_lstm_model_new3.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 3)
]

model = get_model()
model.compile(optimizer=opt,  loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(time_series_generator(train_dir, config, False),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config.num_epochs, validation_steps=validation_steps,
                    validation_data=time_series_generator(val_dir, config, False),
                    callbacks=callbacks)

with K.name_scope(model.optimizer.__class__.__name__):
    for i, var in enumerate(model.optimizer.weights):
        name = 'variable{}'.format(i)
        model.optimizer.weights[i] = tf.Variable(var, name=name)

model.save("../../models/custom_generator", save_format="h5")
