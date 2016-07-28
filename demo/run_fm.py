# from sklearn.datasets import load_svmlight_file
# import numpy as np
# import scipy.sparse as sp
# from pyfm import pylibfm
# from sklearn import metrics
#
#
# train_path = '/data/vcc/adnlog/part-00003'
# test_path = '/data/vcc/adnlog/part-00008'
#
# X_train, y_train = load_svmlight_file(train_path)
# X_test, y_test = load_svmlight_file(test_path)
#
# fm = pylibfm.FM(num_factors=10, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
# # Evaluate
# from sklearn.metrics import log_loss
#
# scores = fm.predict(X_test)
#
# print "Validation log loss: %.4f" % log_loss(y_test,scores)
# fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
# print"Validation log loss: %.4f" % metrics.auc(fpr, tpr)