"""
itterate through an entire directory of videos and split clips into images for training models
"""
import argparse
import numpy as np
import cv2
import os
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help="directory for videos to be split")
ap.add_argument("-o", "--output", required = True,
	help="full path for output file to be stored")
ap.add_argument("-f", "--frames", required = True,
	help="frames per array")
args = vars(ap.parse_args())

def split_clip(in_path, label, filename, frame_array, label_array, frames_per_datapoint):
    vc = cv2.VideoCapture(in_path)
    i = 0
    temp_frame_array = []
    while True:
        grabbed, frame = vc.read()
        if not grabbed:
            break

        if len(temp_frame_array) < frames_per_datapoint:
            temp_frame_array.append(frame)
        else:
            frame_array.append(temp_frame_array)
            label_array.append(label)
            temp_frame_array = []
            temp_frame_array.append(frame)
    vc.release()
    return frame_array, label_array

in_path = str(args["input"])
out_path = str(args["output"])
frames_per_datapoint = int(args["frames"])
directory_contents = os.listdir(in_path)
frame_array = []
label_array = []
for i in directory_contents:
    dir = "%s/%s" % (in_path, i)
    if os.path.isdir(dir):
        arr_of_vids = os.listdir(dir)
        for video in arr_of_vids:
            print("Converting: %s" % video)
            vid_path = "%s/%s/%s" % (in_path, i, video)
            frame_array, label_array = split_clip(vid_path, i, video.split('.')[0], frame_array, label_array, frames_per_datapoint)
            for i in frame_array:
                np.savetxt(out_path + "/dataset.csv", frame_array, delimiter=",")


# output = np.asarray(frame_array)
# np.savetxt(out_path + "/dataset.csv", output, delimiter=",")

with open(out_path+'/label_list.txt', 'w+') as f:
    for label in label_array:
        f.write('%s\n' % label)
