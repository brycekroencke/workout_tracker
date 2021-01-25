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
import json
import time
import sys
import cv2
import os

from AutoWorkoutTracker.main.bar_tracker import process_video

# sys.path.insert(1, '../..')
# from src.utils.workout_tracker_functions import *
# from src.utils.hyperparameters import get_hyperparam_obj
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=80,
                help="max buffer size")
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument('--predict_each_frame', '-f', type=int, default=0,
                help='if 1 predicts the exercise of each frame in video. helpful for debugging.')
args = vars(ap.parse_args())



# if no video is provided use webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])


data = process_video(camera)

print(data)

camera.release()
cv2.destroyAllWindows()
