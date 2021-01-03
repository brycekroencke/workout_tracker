
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
import csv
import os
import cv2
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py

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

class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0, ) + shape,
                maxshape=(None, ) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len, ) + shape)

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()


def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    images =[]
    while success:
        success, image = vidObj.read()

        if image is not None:
            image = cv2.resize(image, (resize_x, resize_y)).astype("float32")
            # image = extract_face_features(detect_face(image))[0]
            images.append(image)

            count += 1
            if count == frame_count:
                result = np.array(images)
                #result = np.array([images[i][0] for i in range(len(images))])
                return result


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


videonames = []
video_di = {}
cats = ["deadlift", "ohp", "squat"]
local ='C:/Users/Bryce/Documents/GitHub/workout_tracker_data/vid_clips_to_train_on/'
labels = []
directory_names = os.listdir(local)
print(directory_names)
# h5f = h5py.File('data.h5', 'a')
shape = (frame_count, resize_x, resize_y, 3)
hdf5_store = HDF5Store('data.h5','X', shape=shape)

# hdf5_store_Y = HDF5Store('data.h5','Y', shape=(1,1))
batch_count = 0
for di in directory_names:
    mypath = local + di + "/"
    subdirectory_names = os.listdir(mypath)
    #
    # grp = h5f.create_group(di)
    for video in subdirectory_names:
        finalpath = mypath + video
        print(video)
        #video_di[video] = pad_video(FrameCapture(finalpath))
        v = pad_video(FrameCapture(finalpath))
        if v.shape[2] != resize_y:
            continue
        else:
            print(v.shape)
        labels.append(cats.index(di))
        #h5f.create_dataset(di + "/" + video, data=v)
        # labels.append(di)
        hdf5_store.append(v)
        #hdf5_store_Y.append(cats.index(di))
        #videonames.append(video)
with open('label_list.csv', 'w+') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(labels)
