from collections import Counter
from datetime import date
import numpy as np
import argparse
import json
import time
import os
import calendar

data = {}


with open("workout_overview/workout_log.json") as f:
    if os.stat("workout_overview/workout_log.json").st_size == 0:
        print('File is empty')
    else:
        print('Loading data')
        data = json.load(f)


today = date.today()
d1 = today.strftime("%d/%m/%Y")

for date in data:
    day_str = list(calendar.day_name)[data[date]["day"]]
    lifts = data[date]["lifts"]
    print("=" * 35)
    print("%s - %s" % (day_str, date))
    print("-" * 35)

    for lift in lifts:
        print(lift)
        for weight in lifts[lift]:
            rep_list = []
            # print("Work done at weight %s" % weight)
            for reps in lifts[lift][weight]:
                rep_list.append(reps["reps"])
            set_rep_count = Counter(rep_list)
            for key in set_rep_count:
                print("%s by %s at %s lbs." % (set_rep_count[key],key, weight))
        print("")
    print("")
