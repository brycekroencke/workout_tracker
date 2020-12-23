"""
itterate through an entire directory of videos and split clips into images for training models
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help="directory for videos to be split")
ap.add_argument("-o", "--output", required = True,
	help="directory for images to be saved")
args = vars(ap.parse_args())

in_path = str(args["input"])
out_path = str(args["output"])
directory_contents = os.listdir(in_path)

for i in directory_contents:
    dir = "%s/%s" % (in_path, i)
    if os.path.isdir(dir):
        print("splitting videos in the %s directory" % i)
        if os.path.isdir("%s/%s" %(out_path, i)):
            arr = os.listdir(dir)
            print(arr)
        else:
            print("creating a %s directory inside of the output folder" % i)
            os.mkdir("%s/%s" % (out_path, i))
            arr = os.listdir(dir)
            print(arr)
