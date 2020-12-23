"""
itterate through an entire directory of videos and split clips into images for training models
"""
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help="directory for videos to be split")
ap.add_argument("-o", "--output", required = True,
	help="directory for images to be saved")
args = vars(ap.parse_args())

def split_clip(inpath, outpath, label, filename):
    vc = cv2.VideoCapture(inpath)
    i = 0
    while True:
        grabbed, frame = vc.read()
        if not grabbed:
            break
        while os.path.exists("%s/%s/%s-%s.jpg" % (outpath, label, filename, i)):
            i += 1
        cv2.imwrite("%s/%s/%s-%s.jpg" % (outpath, label, filename, i), frame)
    vc.release()
    return



in_path = str(args["input"])
out_path = str(args["output"])
directory_contents = os.listdir(in_path)

for i in directory_contents:
    dir = "%s/%s" % (in_path, i)
    if os.path.isdir(dir):
        print("splitting videos in the %s directory" % i)
        if not os.path.isdir("%s/%s" %(out_path, i)):
            print("creating a %s directory inside of the output folder" % i)
            os.mkdir("%s/%s" % (out_path, i))
        arr_of_vids = os.listdir(dir)
        for video in arr_of_vids:
            print("Splitting: %s" % video)
            vid_path = "%s/%s/%s" % (in_path, i, video)
            split_clip(vid_path, out_path, i, video.split('.')[0])
