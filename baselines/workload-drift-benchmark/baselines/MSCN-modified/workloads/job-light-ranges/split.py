import numpy as np
fp = open("adaptation_2_prediction.csv")
lines = fp.readlines()
labels = []
for line in lines:
    labels.append(int(line.split("#")[-1]))
median = np.median(labels)
fpout1 = open("small_label_queries.csv", 'w')
fpout2 = open("large_label_queries.csv", "w")
for line in lines:
    if int(line.split('#')[-1]) < median:
        fpout1.write(line)
    else:
        fpout2.write(line)

