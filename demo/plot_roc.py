import sys

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import log_loss


# import matplotlib.pyplot as plt
# import csv




def main(argv):
    label_file = argv[0]
    score_file = argv[1]
    output_file = argv[2]
    # load csv file
    labels = []
    # with open('/storage/phuongdv/vc-data/te.csv') as csvfile:
    #     reader = csv.DictReader(csvfile, delimiter=' ')
    #     for row in reader:
    #         labels.append(int(row["Label"]))

    with open(label_file) as csvfile:
        for line in csvfile:
            row = line.split(' ')
            labels.append(int(row[0]))

    lines = [line.rstrip('\n') for line in open(score_file)]
    scores = []
    for line in lines:
        scores.append(float(line))

    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    log_loss_ffm = log_loss(labels, scores)

    print("log loss : {}".format(log_loss_ffm))
    print("AUC : {}".format(roc_auc))
    print("PR|AUC: {}".format(average_precision_score(labels, scores)))
    # write to a files

    # outfile = open(output_file, 'w')
    # outfile.write("label scores\n")
    # for i in range(0, len(labels)):
    #     outfile.write("{} {}\n".format(labels[i], scores[i]))
    #
    # outfile.close()


if __name__ == '__main__':
    main(sys.argv[1:])
