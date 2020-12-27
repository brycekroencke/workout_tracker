"""
Trains the classifier model using a custom dataset of images
Classifies lifting movements for bar_tracker.py
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import matplotlib
import argparse
import pickle
import time
import sys
import cv2
import os

# Currently, memory growth needs to be the same across GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
tf.get_logger().setLevel('WARNING')
print(tf.__version__)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
    help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
    help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
    help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

lr = 0.0001
batch_size = 128
categories = ["deadlift", "ohp", "squat"]
train_data_dir = args["dataset"]
img_height, img_width = 224,224

train_datagen = ImageDataGenerator(
    rotation_range=10,
	zoom_range=0.1,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest",
    # rescale=1./255,
    validation_split=0.2,
) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
) # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
) # set as validation data

# test_datagen = ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     color_mode="rgb",
#     shuffle = False,
#     class_mode='categorical',
#     subset='test',
#     batch_size=1)

# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
# plt.style.use("ggplot")
# for i in range(len(train_generator)):
#     batch = train_generator[i]
#
#     x, y = batch
#
#
#
#     print('images in batch:', len(x))
#
#     for image in x:
#         print(image)
#         imgplot = plt.imshow((image).astype(np.uint8))
#         plt.show()

baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(categories), activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False


# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(learning_rate=lr,name='Adam')
#opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = args["epochs"],
    callbacks=[es]
)

epochs_prior_to_finetune = len(history.history['loss'])
print("early stopping finished at epoch: %s" % str(epochs_prior_to_finetune))
# serialize the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

#
# for layer in baseModel.layers:
#     layer.trainable = True
#
# filenames = test_generator.filenames
# nb_samples = len(filenames)
#
# test_labels=test_generator.classes
# predictions = model.predict_generator(test_generator,steps = nb_samples)
# y_pred = np.argmax(predictions, axis=-1)
# # print(classification_report(test_labels, y_pred))
#
#
#
# n_batches = len(test_generator)
#
# confusion_matrix(
#     np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(n_batches)]),
#     np.argmax(model.predict_generator(test_generator, steps=batch_size), axis=1)
# )
#
# #
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(x=testX.astype("float32"), batch_size=batch_size)
# print(classification_report(testY.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=categories))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.style.use("ggplot")
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('plot.png')
# plt.show()


baseModel.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(baseModel.layers))
# Fine-tune from this layer onwards
fine_tune_at = 100
# Freeze all the layers before the `fine_tune_at` layer
for layer in baseModel.layers[:fine_tune_at]:
  layer.trainable =  False


model.compile(loss="categorical_crossentropy",
              optimizer =  tf.keras.optimizers.Adam(learning_rate=(lr/10),name='Adam'),
              metrics=['accuracy'])
model.summary()

fine_tune_epochs = 5
initial_epochs = epochs_prior_to_finetune
total_epochs =  initial_epochs + fine_tune_epochs



history_fine = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[es]

)

print("[INFO] serializing network...")
model.save(args["model"] + "_finetuned", save_format="h5")


print("early stopping for fine tuning finished at epoch: %s" % len(history_fine.history['loss']))

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.style.use("ggplot")
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('plot_ft.png')
# plt.show()

# serialize the label binarizer to disk
# f = open(args["label_bin"], "wb")
# f.write(pickle.dumps(lb))
# f.close()
