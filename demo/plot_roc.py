from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#import matplotlib.pyplot as plt
#import csv

# load csv file
labels = []
# with open('/storage/phuongdv/vc-data/te.csv') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=' ')
#     for row in reader:
#         labels.append(int(row["Label"]))

with open('/storage/phuongdv/vcc-ffm/te.ffm') as csvfile:
    for line in csvfile:
        row = line.split(' ')
        labels.append(int(row[0]))

lines = [line.rstrip('\n') for line in open('/storage/phuongdv/vcc-ffm/te.out.cal')]
scores = []
for line in lines:
    scores.append(float(line))


# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

print(roc_auc)