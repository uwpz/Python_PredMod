


# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm


# Load from dump
with open("data/info_20000_1024_einheit.pkl", "rb") as file:
    pick = pickle.load(file)
yhat = pick["yhat"]
labels = pick["labels"]
df_y = pick["df_y"]

# Get yhat and y related information
y_files = df_y["y_files"].values
y_true = df_y["y_true"].values
y_pred = df_y["y_pred"].values
y_true_label = df_y["y_true_label"].values
y_pred_label = df_y["y_pred_label"].values
yhat_true = df_y["yhat_true"].values
yhat_pred = df_y["yhat_pred"].values


plot_all_performances(y=y_true, yhat=yhat, labels=labels, target_type="MULTICLASS", colors=colors, ylim=None,
                      w=18, h=12, pdf="blub.pdf")




import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=1000, random_state=42, cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)
