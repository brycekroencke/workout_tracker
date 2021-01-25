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

"""Return all local y mins in list of points"""
def local_min(ys):
    return [y[1] for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1][1] >= y[1]))
            and ((i == len(ys) - 1) or (y[1] < ys[i+1][1]))]


"""Return all local y maxs in list of points"""
def local_max(ys):
    return [y[1] for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1][1] < y[1]))
            and ((i == len(ys) - 1) or (y[1] >= ys[i+1][1]))]


"""
    Count the number of reps in a list and increment sets
    Uses local mins and maxs to determine number of reps
    Ignores small vertical fluctuations
"""
def count_reps(arr):
    rep_counter = 0
    if arr:
        local_mins = local_min(arr)
        local_maxs = local_max(arr)

        if len(local_maxs) > len(local_mins):
            for i in range(len(local_maxs)-1):
                if abs(local_maxs[i] - local_mins[i]) > 30:
                    rep_counter += 1

        elif len(local_maxs) < len(local_mins):
            for i in range(len(local_mins)-1):
                if abs(local_maxs[i] - local_mins[i]) > 30:
                    rep_counter += 1
        else:
            for i in range(len(local_maxs)):
                if abs(local_maxs[i] - local_mins[i]) > 30:
                    rep_counter += 1
    return rep_counter


def count_sets(label, sets):
    if label in sets.keys():
        sets[label] = sets[label] + 1
    else:
        sets[label] = 1
    return sets[label]



"""
    Return True if all points in list are close to eachother (close to no movement)
    else return False
"""
def is_still(arr, stillness_threshold = 3):
    for i in range(len(arr)-1):
        if abs(arr[i][0] - arr[i+1][0]) > stillness_threshold or abs(arr[i][1] - arr[i+1][1]) > stillness_threshold:
            return False
    return True

def horizontal_movement(arr, threshold = 30):
    net_horizontal_movement = 0
    for i in range(int((len(arr)-1) / 3)):
        net_horizontal_movement += arr[i][0] - arr[i+1][0]

    if abs(net_horizontal_movement) > threshold:
        return True
    else:
        return False

def vertical_movement(arr, threshold = 30):
    net_vertical_movement = 0
    for i in range(int((len(arr)-1) / 3)):
        net_vertical_movement += arr[i][1] - arr[i+1][1]

    if abs(net_vertical_movement) > threshold:
        return True
    else:
        return False

def init_workout():
    today = date.today()
    d1 = today.strftime("%m/%d/%Y")
    data = {}
    with open("../../workout_logs/workout_log.json", 'r+') as f:
        if os.stat("../../workout_logs/workout_log.json").st_size == 0:
            print('File is empty')
        else:
            data = json.load(f)
    #if date not in json init data
    if d1 not in data.keys():
        lift_data = {}
        data[d1] = {
            'day': today.weekday(),
            'lifts': lift_data
        }
    return data, d1

def update(arr, label, sets, data, d1, output_list, labels):
    reps = count_reps(arr)
    if reps != 0:
        counter = Counter(labels)
        label = counter.most_common(1)[0][0]
        count_sets(label, sets)
        print("%s   set: %d reps: %d" % (label, sets[label], reps))
        output_list.append([label, sets[label], reps])
        weight = 200
        if label in data[d1]["lifts"].keys():
            if weight in data[d1]["lifts"][label].keys():
                data[d1]["lifts"][label][weight].append({
                            'reps': reps ,
                            })
            else:
                data[d1]["lifts"][label][weight] = [{
                    'reps': reps ,
                }]

        else:
            data[d1]["lifts"][label] = {}
            data[d1]["lifts"][label][weight] = [{
                        'reps': reps,
            }]



def update_json(data):
    with open("../../workout_logs/workout_log.json", 'w') as f:
        json.dump(data, f, indent=4)


def predict_frame(model, frame_buffer, labels, config):
    preds = model.predict(np.expand_dims(frame_buffer, axis=0))[0]
    label = config.categories[np.argmax(preds)]
    labels.append(label)
    print(preds, label)
    return label

def add_label_and_fps(frame, fps_time, label):
    fps_text = "{:.1f}".format(1.0 / (time.time() - fps_time), 1)
    cv2.putText(frame,
                "FPS: %s" % (fps_text),
                (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    label_text = "activity: {}".format(label)
    cv2.putText(frame, label_text, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    return frame


def append_to_frame_buffer(frame_buffer, frame, config):
    frame_buffer = deque(frame_buffer)
    output = frame.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = cv2.resize(
        output, (config.width, config.height)).astype("float32")
    # output /= 255.
    frame_buffer.appendleft(output)
    if len(frame_buffer) > config.seq_len:
        frame_buffer.pop()
    frame_buffer = np.array(frame_buffer)
    return frame_buffer

def track_point(frame, config, pts):
    # define the lower and upper boundaries of the tracked obj color
    # (H/2, (S/100) * 255, (V/100) * 255)
    # pink highlighter
    colorLower = np.array([170, 130, 140])
    colorUpper = np.array([178, 255, 255])
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
    return pts
