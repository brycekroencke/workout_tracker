import os
import re
import cv2
import sys
import glob
import random
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.utils import to_categorical

#augmentations to be performed on the timeseries dataset
seq_img = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=True),
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20), order=1, mode="edge"),
    #iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, mode="edge")
])

def split_num(s):
    return list(filter(None, re.split(r'(\d+)', s)))

def augment_timeseries_imgs(images, seq):
    seq_img_i = seq.to_deterministic()
    aug_time_series = [seq_img_i.augment_image(frame) for frame in images]
    return aug_time_series

def time_series_generator(img_dir, config, aug = True):
    #A generator that returns batch of array of images plus a label
    dirs = glob.glob(img_dir + "/*")
    dirs2 = []
    for dir in dirs:
        dirs2.append(dir.replace('\\', '/'))
    random.shuffle(dirs2)
    counter = 0
    while True:
        input_images = np.zeros((config.batch_size, config.seq_len, config.width, config.height, 3))
        labels = np.zeros((config.batch_size, 3))
        if (counter + config.batch_size >= len(dirs2)):
            counter = 0
        for i in range(config.batch_size):
            input_imgs = glob.glob(dirs2[counter + i] + "/img_*.jpg")
            imgs = []
            input_imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
            for img in input_imgs:
                temp = cv2.imread(img)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                temp = cv2.resize(temp, (config.height, config.width))
                imgs.append(np.asarray(temp))

            input_images[i] = imgs
            label = split_num(dirs2[counter + i].split('/')[-1])[0]
            labels[i] = to_categorical(config.categories.index(label), len(config.categories))
            input_images[i] /= 255.
            if aug:
                input_images[i] = augment_timeseries_imgs(input_images[i], seq_img)
        yield (input_images, labels)
        counter += config.batch_size
