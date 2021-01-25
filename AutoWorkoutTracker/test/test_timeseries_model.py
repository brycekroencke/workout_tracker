import os
import numpy as np
from tensorflow.keras.models import load_model
import sys

sys.path.insert(1, '../..')
from AutoWorkoutTracker.generator.custom_generator import time_series_generator
from AutoWorkoutTracker.utils.hyperparameters import get_hyperparam_obj


train_dir = '/Users/Bryce/Documents/GitHub/workout_tracker_data/train'

categories = ["deadlift", "ohp", "squat"]es

config = get_hyperparam_obj()
config.batch_size = 64
gen = time_series_generator(train_dir, config)
videos, label = next(gen)
v = videos
l = label

model = load_model("../../models/custom_generator", compile=False)
preds = model.predict(v)
for i in range(len(preds)):
    p = np.argmax(preds[i])
    a = int(l[i][0])
    if a == p:
        print("True: %d %d" % (p, a))
    else:
        print("False: %d %d" % (p, a))
