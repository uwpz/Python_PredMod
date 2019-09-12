


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
