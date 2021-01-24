"""
itterate through an entire directory of videos and split clips into images for training models
"""


import argparse
import shutil
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="directory for videos to be split")
ap.add_argument("-o", "--output", required=True,
                help="directory for images to be saved")
ap.add_argument("-f", "--frames", required=True,
                help="frames per directory")
args = vars(ap.parse_args())

width, height = 128, 128




def split_clip(inpath, outpath, label, filename, frames_per_dir):
    vc = cv2.VideoCapture(inpath)
    i = 0
    j = 0
    frame_count = -1
    while True:
        grabbed, frame = vc.read()
        if not grabbed:
            break

        frame_count += 1
        if frame_count >= frames_per_dir:
            j += 1
            i = 0
            frame_count = 0

        if not os.path.isdir("%s/%s-%s" % (out_path, filename, j)):
            os.mkdir("%s/%s-%s" % (out_path, filename, j))

        # while os.path.exists("%s/%s/%s-%s/img_%s.jpg" % (outpath, label, filename, j, i)):
        #     i += 1

        while os.path.exists("%s/%s-%s/img_%s.jpg" % (outpath, filename, j, i)):
            i += 1

        frame = cv2.resize(frame, (width, height))
        cv2.imwrite("%s/%s-%s/img_%s.jpg" %
                    (outpath, filename, j, i), frame)
    vc.release()
    return


in_path = str(args["input"])
out_path = str(args["output"])
frames_per_dir = int(args["frames"])
directory_contents = os.listdir(in_path)

for i in directory_contents:
    dir = "%s/%s" % (in_path, i)
    if os.path.isdir(dir):
        print("splitting videos in the %s directory" % i)
        # if not os.path.isdir("%s/%s" % (out_path, i)):
        #     print("creating a %s directory inside of the output folder" % i)
        #     os.mkdir("%s/%s" % (out_path, i))
        arr_of_vids = os.listdir(dir)
        for video in arr_of_vids:
            print("Splitting: %s" % video)
            vid_path = "%s/%s/%s" % (in_path, i, video)
            split_clip(vid_path, out_path, i,
                       video.split('.')[0], frames_per_dir)


all_train_dirs = os.listdir(out_path)
for i in all_train_dirs:
    if len(os.listdir("%s/%s" % (out_path , i))) < frames_per_dir:
        try:
            shutil.rmtree("%s/%s" % (out_path , i))
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
