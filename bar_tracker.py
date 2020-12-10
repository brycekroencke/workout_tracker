#track the users face position and barbell postion
from tensorflow.keras.models import load_model
from operator import itemgetter
from collections import deque
from datetime import date
import numpy as np
import argparse
import imutils
import pickle
import json
import time
import cv2

from collections import Counter
# import logging


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

"""
    Return True if all points in list are close to eachother (close to no movement)
    else return False
"""
def is_still(arr, stillness_threshold = 5):
    for i in range(len(arr)-1):
        if abs(arr[i][0] - arr[i+1][0]) > stillness_threshold or abs(arr[i][1] - arr[i+1][1]) > stillness_threshold:
            return False
    return True



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=30,
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
args = vars(ap.parse_args())

# logger = logging.getLogger('TfPoseEstimator-WebCam')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
fps_time = 0
today = date.today()
d1 = today.strftime("%d/%m/%Y")
data = {}
with open("workout_overview/workout_log.json") as f:
    one_char = f.read(1)
    # if not fetched then file is empty
    if one_char:
        data = json.load(f)

#data[d1] = {}

if d1 not in data.keys():
    print("ADDING DATE TO FILE")
    lift_data = []
    data[d1] ={
        'day': today.weekday(),
        'lifts': lift_data
    }
else:
    print("APPENDING TO DATE")
#
# logger.debug('initialization %s : %s' % (args["model2"], get_graph_path(args["model2"])))
# w, h = model_wh(args["resize"])
# if w > 0 and h > 0:
#     e = TfPoseEstimator(get_graph_path(args["model2"]), target_size=(w, h), trt_bool=str2bool(args["tensorrt"]))
# else:
#     e = TfPoseEstimator(get_graph_path(args["model2"]), target_size=(432, 368), trt_bool=str2bool(args["tensorrt"]))
# logger.debug('cam read+')
# cam = cv2.VideoCapture(args.camera)
# ret_val, image = cam.read()
# logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

# load the trained model and label binarizer from disk
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])


pts = deque(maxlen=args["buffer"])


all_ys = []
output_list = []
sets = 0
label = ""
labels = []
wait_for_movement = False

# define the lower and upper boundaries of the tracked obj color
# yellow highlighter
# colorLower = (24, 100, 100)
# colorUpper = (44, 255, 255)

colorLower = (20, 100, 125)
colorUpper = (100, 255, 255)

#pink highlighter
# colorLower = (190, 30, 150)
# colorUpper = (255, 110, 250 )

#blue sticky color

# colorLower = (100, 160, 160)
# colorUpper = (200, 250, 250)

# #attempt at finding barbell color
# colorLower = (0, 1, 240)
# colorUpper = (360, 4, 254)


if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

	#reached end of video
    if args.get("video") and not grabbed:
        break


    frame = imutils.resize(frame, width=432, height=368)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
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
                print("%s   set: %d reps: %d" % (label, sets, reps))
                output_list.append([label, sets, reps])
                weight = 200
                print(Counter(labels))
                data[d1]["lifts"].append({label :{
                            'reps': reps ,
                            'sets': sets,
                            'weight': weight,
                }})
            labels = []
            all_ys = []
            wait_for_movement = True
    else:
        #Object is moving only predict when obj is moving
        output = frame.copy()
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = cv2.resize(output, (224, 224)).astype("float32")
        output -= mean

        # make predictions on the frame and then update the predictions
        # queue
        # e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(368, 368))
        # humans = e.inference(output)
        # frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)


        preds = model.predict(np.expand_dims(output, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        labels.append(label)
        wait_for_movement = False


    # loop over the set of tracked points
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
        	continue
        # draw connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # # show the frame to our screen
    # humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args["resize_out_ratio"])
    #
    # logger.debug('postprocess+')
    # frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
    #
    # logger.debug('show+')
    cv2.putText(frame,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    fps_time = time.time()
    key = cv2.waitKey(1) & 0xFF

    # press q to exit loop
    if key == ord("q"):
        # print(output_list)
        # today = date.today()
        # d1 = today.strftime("%d/%m/%Y")
        # # data = {}
        # # data[d1] = {}
        # weight = 200
        # lift_data = {"squats" :{
        #             'reps': reps ,
        #             'sets': sets,
        #             'weight': weight,
        # }}
        #

        with open("workout_overview/workout_log.json", 'w') as f:
            json.dump(data, f, indent=4)
        break


camera.release()
cv2.destroyAllWindows()
