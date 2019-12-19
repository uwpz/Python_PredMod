


# General libraries, parameters and functions
import sys
import os
sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm
from initialize import *

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters
n_jobs = 4
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    metric = "auc"  # metric for peformance comparison
else:
    metric = "spear"

# Load results from exploration
df = metr_standard = cate_standard = metr_binned = cate_binned = metr_encoded = cate_encoded = target_labels = None
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")


plot_distr(df, features = np.column_stack((metr, metr + "_BINNED")).ravel(), target_type = TARGET_TYPE,
           varimp = varimp_metr, color = color, ylim = ylim,
           ncol = 4, nrow = 2, w = 24, h = 18, pdf = None)