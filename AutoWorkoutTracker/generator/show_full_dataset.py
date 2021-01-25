import cv2
import os
import shutil
import glob
import re


def split_num(s):
    return list(filter(None, re.split(r'(\d+)', s)))

def show_dataset(img_dir):
    #A generator that returns batch of array of images plus a label
    dirs = glob.glob(img_dir + "/*")
    dirs2 = []
    for dir in dirs:
        dirs2.append(dir.replace('\\', '/'))
    for dir in dirs2:
        input_imgs = glob.glob(dir + "/img_*.jpg")
        label = split_num(dir.split('/')[-1])[0]
        display = True
        input_imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
        for img in input_imgs:
            temp = cv2.imread(img)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (512, 512)).astype("float32")
            temp /= 255.
            text = "activity: {}".format(label)
            cv2.putText(temp, text, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            cv2.imshow('Frame', temp[:,:,::-1])

            if cv2.waitKey(2) & 0xFF == ord('d'):
                print(dir)
                try:
                    shutil.rmtree(dir)
                except OSError as e:
                    print("Error: %s : %s" % (dir, e.strerror))
                break
            # Press Q on keyboard to  exit
            if cv2.waitKey(2) & 0xFF == ord('q'):
                display = False
                break
        if not display:
            break

show_dataset('/Users/Bryce/Documents/GitHub/workout_tracker_data/train')
