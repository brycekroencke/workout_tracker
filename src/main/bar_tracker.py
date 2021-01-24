"""
The primary program which tracks the barbell and classifys the users exercise movements using tensorflow
User can input a videofile or get realtime classification using the webcam
All recorded data is saved to a json file
"""

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

sys.path.insert(1, '../..')
from src.utils.workout_tracker_functions import *
from src.utils.hyperparameters import get_hyperparam_obj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

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


config = get_hyperparam_obj()

data, d1 = init_workout()
model = load_model(args["model"])
pts = deque(maxlen=args["buffer"])
fps_time = 0
all_ys, output_list, labels = [], [], []
sets = {}
label = ""
wait_for_movement = True
first_pred = True

# define the lower and upper boundaries of the tracked obj color
# (H/2, (S/100) * 255, (V/100) * 255)
# pink highlighter
colorLower = np.array([170, 130, 140])
colorUpper = np.array([178, 255, 255])

# if no video is provided use webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])


first_pass = True
frame_buffer = deque([])
while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:  # reached end of video
        update(all_ys, label, sets, data, d1, output_list, labels)
        update_json(data)
        break

    frame_buffer = deque(frame_buffer)
    output = frame.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = cv2.resize(
        output, (config.width, config.height)).astype("float32")
    output /= 255.
    frame_buffer.appendleft(output)
    if len(frame_buffer) > config.seq_len:
        frame_buffer.pop()
    frame_buffer = np.array(frame_buffer)

    if first_pass and len(frame_buffer) == 64:
        print(frame_buffer.shape)
        label = predict_frame(model, frame_buffer, labels, config)
        first_pass = False

    frame = imutils.resize(frame, width=432, height=368)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color and remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the barbell
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        center = (cX, cY)

        # draw points if radius is large enough
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
    pts.appendleft(center)

    # keep track of all non None points and trim off excess horizontal movement
    if pts[0] is not None and pts[1] is not None:
        if abs(pts[0][0] - pts[1][0]) < abs(pts[0][1] - pts[1][1]) and abs(pts[0][1] - pts[1][1]) < 3:
            all_ys.append(pts[0])

    pts_no_nones = [i for i in pts if i]

    if not all(v is None for v in pts):

        if (is_still(pts_no_nones)):
            #Object is still
            if not wait_for_movement:
                update(all_ys, label, sets, data, d1, output_list, labels)
                labels = []
                all_ys = []
                wait_for_movement = True
        else:
            # Object is moving only predict when obj is moving
            if not args["predict_each_frame"] and vertical_movement(pts_no_nones) and not horizontal_movement(pts_no_nones) and len(frame_buffer) == config.seq_len:
                label = predict_frame(model, frame_buffer, labels, config)
                if first_pred:
                    for i in range(50):
                        print("\n")
                    print("Logging workout for %s" % (d1))
                    first_pred = False
            wait_for_movement = False

    if args["predict_each_frame"] and len(frame_buffer == 64):
        label = predict_frame(model, frame_buffer, labels, config)
        if first_pred:
            for i in range(50):
                print("\n")
            print("Logging workout for %s" % (d1))
            first_pred = False

    # loop over the set of tracked points and draw connecting lines for tracing
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    frame = add_label_and_fps(frame, fps_time, label)
	fps_time = time.time()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press q to exit loop and update json
    if key == ord("q"):
        update_json(data)
        break


camera.release()
cv2.destroyAllWindows()
