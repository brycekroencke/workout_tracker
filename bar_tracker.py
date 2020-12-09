#track the users face position and barbell postion
from collections import deque
import numpy as np
import argparse
import imutils
from operator import itemgetter
import json
import cv2
from datetime import date
from tensorflow.keras.models import load_model
from collections import deque
import pickle

def local_min(ys):
    return [y[1] for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1][1] >= y[1]))
            and ((i == len(ys) - 1) or (y[1] < ys[i+1][1]))]


def local_max(ys):
    return [y[1] for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1][1] < y[1]))
            and ((i == len(ys) - 1) or (y[1] >= ys[i+1][1]))]


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def count_reps(arr, set_counter):
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
            set_counter += 1
            return (set_counter, rep_counter)
        else:
            return (set_counter, rep_counter)

    else:
        return (set_counter, rep_counter)

def is_still(arr):
    for i in range(len(arr)-1):
        if abs(arr[i][0] - arr[i+1][0]) > 5 or abs(arr[i][1] - arr[i+1][1]) > 5:
            return False
    return True



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=60,
	help="max buffer size")


ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())




# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])



all_ys = []
sets = 0

# define the lower and upper boundaries of the tracked obj color
#yellow highlighter
colorLower = (24, 100, 100)
colorUpper = (44, 255, 255)

#pink highlighter
# colorLower = (170, 50, 50)
# colorUpper = (255, 230, 230 )
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

wait_for_movement = False

while True:
    (grabbed, frame) = camera.read()

	#reached end of video
    if args.get("video") and not grabbed:
        break

    # barbell movement track
    # resize the frame, inverted ("vertical flip" w/ 180degrees),
    # blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    #frame = imutils.rotate(frame, angle=180)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the barbell
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

	# only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
        	# draw the circle and centroid on the frame,
        	# then update the list of tracked points
        	cv2.circle(frame, (int(x), int(y)), int(radius),
        		(0, 255, 255), 2)
        	cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
    pts.appendleft(center)


    #333 = bottom 0 = top

    #keep track of all non None points
    if pts[0] is not None:
        all_ys.append(pts[0])

    pts_no_nones = [i for i in pts if i]
    if is_still(pts_no_nones) and not all(v is None for v in pts) :
        #Object is still
        if wait_for_movement:
            continue
        else:
            sets, reps = count_reps(all_ys, sets)
            if reps != 0:
                print("set: %d reps: %d" % (sets, reps))
            all_ys = []
            wait_for_movement = True
    else:
        #Object is moving only predict when obj is moving
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean

        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]

        print(label)

        wait_for_movement = False


    # #BAR IS STILL OR OFF SCREEN FOR DURATION OF BUFFER
    # if (all(v is None for v in pts) or is_still(pts_no_nones)) and not wait_for_movement:
    #     sets, reps = count_reps(all_ys, sets)
    #     all_ys = []
    #     wait_for_movement = True

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
        	continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press q to exit loop
    if key == ord("q"):
        if len(all_ys) > 0:
            sets, reps = count_reps(all_ys, sets)
            all_ys = []
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        data = {}
        data[d1] = {}
        weight = 200
        lift_data = {"squats" :{
                    'reps': reps ,
                    'sets': sets,
                    'weight': weight,
        }}

        data[d1] ={
            'day': today.weekday(),
            'lifts': lift_data
        }
        with open("workout_overview/workout_log.json", 'w') as f:
            json.dump(data, f, indent=4)
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
