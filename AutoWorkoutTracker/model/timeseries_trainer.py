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
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, GRU, MaxPool2D
from tensorflow.keras.layers import Input, add, Conv2D, Flatten, Dense, Dropout, LSTM, GlobalMaxPool2D


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


sys.path.insert(1, '../..')
from AutoWorkoutTracker.utils.visualizer import visualize_generator
from AutoWorkoutTracker.utils.hyperparameters import get_hyperparam_obj, update_hyperparams
from AutoWorkoutTracker.generator.custom_generator import time_series_generator

hyperparams = {
    "num_epochs": 20,
    "batch_size": 8,
    "height": 224,
    "width": 224,
    "lr": .01,
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

# temp_batch_size = config.batch_size
# config.batch_size = 8
# gen_val = time_series_generator(val_dir, config)
# visualize_generator(gen_val)
# vids, labs = next(gen_val)
# print(vids[0])
# print(vids.dtype)
# print(vids[0].dtype)
# gen_train = time_series_generator(train_dir, config)
# visualize_generator(gen_train)
# vids, labs = next(gen_train)
# print(vids[0])
# print(vids.dtype)
# print(vids[0].dtype)
#
# config.batch_size = temp_batch_size

steps_per_epoch = len(glob.glob(train_dir + "/*")) // config.batch_size
validation_steps = len(glob.glob(val_dir + "/*")) // config.batch_size

print(len(config.categories))

def get_model():
    pretrained_cnn = load_model("../../models/custom_dataset_model3_finetuned")
    pretrained_cnn.summary()
    for layer in pretrained_cnn.layers:
        layer.trainable = False


    new_model = Model(pretrained_cnn.inputs, pretrained_cnn.layers[-8].output)
    pretrained_cnn_no_last_layer = Sequential()
    pretrained_cnn_no_last_layer.add(new_model)
    pretrained_cnn_no_last_layer.add(GlobalMaxPool2D())

    # pretrained_cnn_no_last_layer.summary()
    # then create our final model
    model = Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(pretrained_cnn_no_last_layer, input_shape=(config.seq_len, config.width, config.height, 3)))
    # model.add(TimeDistributed(Flatten()))
        # here, you can also use GRU or LSTM
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    # and finally, we make a decision network
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(config.categories), activation='softmax'))
    # return model

    # input_layer = Input(shape=(config.seq_len, config.width, config.height, 3))
    # curr_layer = TimeDistributed(pretrained_cnn_no_last_layer)(input_layer)
    # # curr_layer = TimeDistributed(Dense(64))(curr_layer)
    # # curr_layer = TimeDistributed(Dense(64))(curr_layer)
    # curr_layer = TimeDistributed(Dense(1))(curr_layer)
    # lstm_out = LSTM(64, return_sequences=False, dropout=.5)(curr_layer)
    # x = Dense(64, activation='relu')(lstm_out)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(.3)(x)
    # main_output = Dense(len(config.categories), activation = 'softmax', name='main_output')(x)
    # model = Model(inputs=input_layer, outputs=main_output)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='../../visual_assets/model.png')
    return model


def build_convnet(shape=(config.width, config.height, 3)):
    momentum = .9
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # model.add(MaxPool2D())
    #
    # model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    # model.add(BatchNormalization(momentum=momentum))
    #
    # model.add(MaxPool2D())
    #
    # model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    # model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def action_model(shape=(config.seq_len, config.width, config.height, 3), nbout=3):
    convnet = build_convnet(shape[1:])

    model = Sequential()
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(LSTM(32, return_sequences=False, dropout=0.3))
    # and finally, we make a decision network
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(.5))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

if config.loss_opt == "SGD":
    opt = tf.keras.optimizers.SGD(lr=config.lr, decay=1e-6, momentum=0.5)

else:
    opt = tf.keras.optimizers.Adam(learning_rate=config.lr,name='Adam')

callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='../../models/cnn_lstm_model_new3.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 3)
]

model = get_model()
#model = action_model((config.seq_len, config.width, config.height, 3), 3)
model.compile(optimizer=opt,  loss='categorical_crossentropy', metrics=['accuracy'])#, run_eagerly=True)

model.fit(time_series_generator(train_dir, config),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config.num_epochs, validation_steps=validation_steps,
                    validation_data=time_series_generator(val_dir, config),
                    callbacks=callbacks)

with K.name_scope(model.optimizer.__class__.__name__):
    for i, var in enumerate(model.optimizer.weights):
        name = 'variable{}'.format(i)
        model.optimizer.weights[i] = tf.Variable(var, name=name)

model.save("../../models/custom_generator", save_format="h5")
