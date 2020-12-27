"""
itterate through an entire directory of videos and split clips into images for training models
"""
import argparse
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help="directory for videos to be split")
ap.add_argument("-o", "--output", required = True,
	help="directory for images to be saved")
ap.add_argument("-f", "--frames", required = True,
	help="frames per array")
args = vars(ap.parse_args())

def split_clip(inpath, outpath, label, filename, frame_array):
    vc = cv2.VideoCapture(inpath)
    i = 0
    temp_frame_array = []
    while True:
        grabbed, frame = vc.read()
        if not grabbed:
            break
        if len(temp_frame_array) < 10:
            temp_frame_array.append(frame)
        else:
            frame_array.append(temp_frame_array)
            temp_frame_array = []
            temp_frame_array.append(frame)
    vc.release()
    return frame_array



in_path = str(args["input"])
out_path = str(args["output"])
directory_contents = os.listdir(in_path)
frame_array = []
for i in directory_contents:
    dir = "%s/%s" % (in_path, i)

    if os.path.isdir(dir):
        print("splitting videos in the %s directory" % i)
        if not os.path.isdir("%s/%s" %(out_path, i)):
            print("creating a %s directory inside of the output folder" % i)
            os.mkdir("%s/%s" % (out_path, i))
        arr_of_vids = os.listdir(dir)
        for video in arr_of_vids:
            frame_array = []
            print("Splitting: %s" % video)
            vid_path = "%s/%s/%s" % (in_path, i, video)
            frame_array = split_clip(vid_path, out_path, i, video.split('.')[0], frame_array)


output = np.asarray(frame_array)
np.savetxt("frame_data.csv", output, delimiter=",")
