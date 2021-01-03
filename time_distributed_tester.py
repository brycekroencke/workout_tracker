# # cnn lstm model

import numpy as np
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

import os
import cv2
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Currently, memory growth needs to be the same across GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
tf.get_logger().setLevel('WARNING')
print(tf.__version__)


resize_x = 100
resize_y = 100
frame_count = 64

def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    images =[]
    while success:
        success, image = vidObj.read()
        if image is not None:
            image = cv2.resize(image, (resize_x, resize_y))#.astype("float32")
            # image = extract_face_features(detect_face(image))[0]
            images.append(image)
            # cv2.imshow("Frame", image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


    result = np.array(images)
    #result = np.array([images[i][0] for i in range(len(images))])
    return result


def pad_video(video):
    if video.shape[0] < frame_count:
        while video.shape[0] < frame_count:
            video_padded = np.concatenate((video, video[-(frame_count-len(video)):]), axis = 0)
            if video_padded.shape[0] == frame_count:
                break
    else:
        video_padded = video[:frame_count]
    return video_padded


def split_num(s):
    return list(filter(None, re.split(r'(\d+)', s)))[0]

model = tf.keras.models.load_model('model/timedistributed_model')

cats = ["deadlift", "ohp", "squat"]

test_vid1 = pad_video(FrameCapture("C:/Users/Bryce/Documents/GitHub/workout_tracker_data/videos/squat2.mov"))
test_vid2 = pad_video(FrameCapture("C:/Users/Bryce/Documents/GitHub/workout_tracker_data/videos/deadlift2.mov"))
test_vid3 = pad_video(FrameCapture("C:/Users/Bryce/Documents/GitHub/workout_tracker_data/videos/ohp3.mov"))
test_vids = np.asarray([test_vid1, test_vid2, test_vid3])
pred = model.predict(test_vids)
for i in pred:
    print(cats[np.argmax(i)])
