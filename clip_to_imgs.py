"""
Splits a video file into images for training.
Args: -v path-to-video -l label-to-be-assigned
"""

import cv2
import imutils
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True,
	help="path to the (optional) video file")
ap.add_argument("-l", "--label", required = True,
	help="label for folder to be assigned")
args = vars(ap.parse_args())


vc = cv2.VideoCapture('videos/'+args["video"])
success = 1
i = 0
label = args["label"]

while success:
	success, frame = vc.read()
	while os.path.exists("custom_dataset/%s/%s-%s.jpg" % (label, label, i)):
		i += 1
	cv2.imwrite("custom_dataset/%s/%s-%s.jpg" % (label, label, i), frame)
vc.release()
