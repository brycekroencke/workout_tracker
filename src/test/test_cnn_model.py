from matplotlib.pyplot import imshow, figure, show, title
from tensorflow.keras.models import load_model
from operator import itemgetter
from collections import Counter
from collections import deque
from datetime import date
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import random
import json
import time
import glob
import sys
import cv2
import os

categories = ["deadlift", "ohp", "squat"]
pretrained_cnn = load_model("../../models/custom_dataset_model3_finetuned")

width, height = 224, 224

correct = 0
total = 0
for i in range(100):
    random_class = random.choice(categories)
    dirs = glob.glob('/Users/Bryce/Documents/GitHub/workout_tracker_data/data/%s/*' % random_class)
    random_img = random.choice(dirs)
    temp = cv2.imread(random_img)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (height, width))
    pred = pretrained_cnn.predict(np.expand_dims(temp, axis=0))[0]
    total += 1
    if (categories[np.argmax(pred)] == random_class):
        print("TRUE predicted: %s actual: %s" % (categories[np.argmax(pred)], random_class))
        correct += 1
    else:
        print("FALSE predicted: %s actual: %s" % (categories[np.argmax(pred)], random_class))
    imshow(temp)
    title(random_img.split('/')[-1]);
    show()
print((correct * 1.0)/total)
