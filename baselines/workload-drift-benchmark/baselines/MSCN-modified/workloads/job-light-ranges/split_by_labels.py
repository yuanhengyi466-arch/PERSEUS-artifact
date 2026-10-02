import numpy as np
import os
import sys

fp = open("job-light_equal_distribution_10000.csv")
lines = fp.readlines()
labels = []
for line in lines:
    label = int(line.split('#')[-1]) 
    labels.append(label)
median_label = np.median(labels)
minlabel = np.min(labels)
maxlabel = np.max(labels)
folder_path = "varying_label_queries"
if os.path.exists(folder_path) is False:
    os.makedirs(folder_path)

smallfp = open("{}/small_label_queries.csv".format(folder_path), 'w')
largefp = open("{}/large_label_queries.csv".format(folder_path), 'w')
for line in lines:
    label = int(line.split("#")[-1])
    if label < median_label:
        smallfp.write(line)
        if label == minlabel:
            largefp.write(line)
    else:
        largefp.write(line)
        if label == maxlabel:
            smallfp.write(line)
