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
print("*" * 30)
print("python version: %s" % str(sys.version))
print("tf version: %s" % str(tf.__version__))
print("*" * 30)
#workaround to gpu allocation issue
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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

"""Return True if all points in list are equal else False"""
def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

"""
    Count the number of reps in a list and increment sets
    Uses local mins and maxs to determine number of reps
    Ignores small vertical fluctuations
"""
def count_reps(arr):
    rep_counter = 0
    if len(arr) != 0:
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

        if rep_counter != 0:
            return rep_counter
        else:
            return rep_counter

    else:
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
    with open("workout_overview/workout_log.json", 'r+') as f:
        if os.stat("workout_overview/workout_log.json").st_size == 0:
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
    with open("workout_overview/workout_log.json", 'w') as f:
        json.dump(data, f, indent=4)

def predict_frame(frame, mean, labels):
    output = frame.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = cv2.resize(output, (224, 224)).astype("float32")
    output -= mean


    preds = model.predict(np.expand_dims(output, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    labels.append(label)
    text = "activity: {}".format(label)
    cv2.putText(frame, text, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 1,
    (0, 255, 0), 2)
    return label


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=80,
	help="max buffer size")
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument('--model2', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-s", "--size", type=int, default=30,
	help="size of queue for averaging")
ap.add_argument('--tensorrt', type=str, default="False",
                    help='for tensorrt process.')
ap.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
ap.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')
ap.add_argument('--predict_each_frame', '-f', type=int, default=0,
                    help='if 1 predicts the exercise of each frame in video. helpful for debugging.')
args = vars(ap.parse_args())

data, d1 = init_workout()
# load the trained model and label binarizer from disk
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])
pts = deque(maxlen=args["buffer"])

fps_time = 0
all_ys, output_list, labels = [], [], []
sets = {}
label = ""
wait_for_movement = True

# define the lower and upper boundaries of the tracked obj color
# (H/2, (S/100) * 255, (V/100) * 255)
#yellow highlighter
# colorLower = (24, 100, 100)
# colorUpper = (44, 255, 255)

#pink highlighter
colorLower = np.array([170, 130, 140])
colorUpper = np.array([178, 255, 255])

#if no video is provided use webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

print("\n" * 15)
print("Workout Tracker has been initalized.")

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed: #reached end of video
        update(all_ys, label, sets, data, d1, output_list, labels)
        update_json(data)
        break


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


    #keep track of all non None points and trim off excess horizontal movement
    if pts[0] is not None and pts[1] is not None:
        if abs(pts[0][0] - pts[1][0]) < abs(pts[0][1] - pts[1][1]) and abs(pts[0][1] - pts[1][1]) < 3:
            all_ys.append(pts[0])


    pts_no_nones = [i for i in pts if i]

    if  not all(v is None for v in pts):

        if (is_still(pts_no_nones)):
            #Object is still
            if  not wait_for_movement:
                update(all_ys, label, sets, data, d1, output_list, labels)
                labels = []
                all_ys = []
                wait_for_movement = True
        else:
            #Object is moving only predict when obj is moving
            if not args["predict_each_frame"] and vertical_movement(pts_no_nones) and not horizontal_movement(pts_no_nones):
                label = predict_frame(frame, mean, labels)

            wait_for_movement = False


    if args["predict_each_frame"]:
        label = predict_frame(frame, mean, labels)


    # loop over the set of tracked points and draw connecting lines for tracing
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
        	continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # add fps tracker to frame
    f = "{:.1f}".format(1.0 / (time.time() - fps_time), 1)
    cv2.putText(frame,
                "FPS: %s" % (f),
                (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    fps_time = time.time()
    key = cv2.waitKey(1) & 0xFF

    # press q to exit loop and update json
    if key == ord("q"):
        update_json(data)
        break


camera.release()
cv2.destroyAllWindows()
