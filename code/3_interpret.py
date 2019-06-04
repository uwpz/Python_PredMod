
# ######################################################################################################################
#  Libraries
# ######################################################################################################################

# --- Load general libraries  -------------------------------------------------------------------------------------
# Data
import numpy as np
import pandas as pd
from dill import (load_session, dump_session)

# Plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# plt.ioff(); plt.ion()  # Interactive plotting? ion is default

# Util
import pdb  # pdb.set_trace()  #quit with "q", next line with "n", continue with "c"
from os import getcwd

# My
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from myfunc import *


# --- Load specific libraries  -------------------------------------------------------------------------------------
# ML
import xgboost as xgb


# --- Load results, run 0_init  -----------------------------------------------
exec(open("./code/0_init.py").read())


# ######################################################################################################################
# Initialize
# ######################################################################################################################

# Set metric for peformance comparison
metric = "roc_auc"

# Tuning parameter to use (for xgb)
n_estimators = 1100
learning_rate = 0.01
max_depth = 3
min_child_weight = 10
colsample_bytree = 0.7
subsample = 0.7
gamma = 0


# --- Sample data ----------------------------------------------------------------------------------------------------
# Training data: Just take data from train fold (take all but n_maxpersample at most)
df.loc[df["fold"] == "train", "target"].describe()
df_train, b_sample, b_all = undersample_n(df.loc[df["fold"] == "train", :], 500)
print(b_sample, b_all)

# Test data
df_test = df.loc[df["fold"] == "test"]  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Folds for crossvalidation
split_my5fold = TrainTestSep(5, "bootstrap")

# Check
for i_train, i_test in split_my5fold.split(df):
    print("TRAIN-fold:", df["fold"][i_train].value_counts())
    print("TEST-fold:", df["fold"][i_test].value_counts())
    print("##########")


# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------
# Fit
X_train = create_sparse_matrix(df_train, metr, cate, df_ref=df)
clf = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, min_child_weight=min_child_weight,
                        colsample_bytree=colsample_bytree, subsample=subsample,
                        gamma=0)
fit = clf.fit(X_train, df_train["target"])

# Predict
X_test = create_sparse_matrix(df_test, metr, cate, df_ref=df)
yhat_test = scale_predictions(fit.predict_proba(X_test), b_sample, b_all)
pd.DataFrame(yhat_test).describe()

# Plot performance
plot_all_performances(df_test["target"], yhat_test, pdf=plotloc + "performance.pdf")


# --- Check performance for crossvalidated fits ---------------------------------------------------------------------
d_cv = cross_validate(clf,
                      create_sparse_matrix(df, metr, cate), df["target"],
                      cv=split_my5fold.split(df),  # special 5fold
                      scoring=metric, n_jobs=5,
                      return_estimator=True)
# Performance
print(d_cv["test_score"])


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------
# Variable importance (on train data!)
df_varimp_train = calc_varimp_by_permutation(df_train, df, fit, "target", metr, cate, b_sample, b_all)

# Top features (importances sum up to 95% of whole sum)
features_top = df_varimp_train.loc[df_varimp_train["importance_cum"] < 95, "feature"].values

# Fit again only on features_top
X_train_top = create_sparse_matrix(df_train, metr[np.in1d(metr, features_top)], cate[np.in1d(cate, features_top)],
                                   df_ref=df)
fit_top = clone(clf).fit(X_train_top, df_train["target"])

# Plot performance
X_test_top = create_sparse_matrix(df_test, metr[np.in1d(metr, features_top)], cate[np.in1d(cate, features_top)],
                                  df_ref=df)
yhat_top = scale_predictions(fit_top.predict_proba(X_test_top), b_sample, b_all)
plot_all_performances(df_test["target"], yhat_top, pdf=plotloc + "performance_top.pdf")


# ######################################################################################################################
# Diagnosis
# ######################################################################################################################

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO


# ######################################################################################################################
# Variable Importance
# ######################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------
xgb.plot_importance(fit)


# --- Variable Importance by permuation argument -------------------------------------------------------------------
# Importance for "total" fit (on test data!)
df_varimp = calc_varimp_by_permutation(df_test, df, fit, "target", metr, cate, b_sample, b_all)
topn = 5
topn_features = df_varimp["feature"].values[range(topn)]

# Add other information (e.g. special category): category variable is needed -> fill with at least with "dummy"
df_varimp["Category"] = pd.cut(df_varimp["importance"], [-np.inf, 10, 50, np.inf], labels=["low", "medium", "high"])

# Crossvalidate Importance: ONLY for topn_vars
df_varimp_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(split_my5fold.split(df)):
    df_tmp = calc_varimp_by_permutation(df.iloc[i_train, :], df, d_cv["estimator"][i],
                                        "target", metr, cate,
                                        b_sample, b_all,
                                        features=topn_features)
    df_tmp["run"] = i
    df_varimp_cv = df_varimp_cv.append(df_tmp)


# Plot
fig, ax = plt.subplots(1, 1)
sns.barplot("importance", "feature", hue="Category", data=df_varimp.iloc[range(topn)],
            dodge=False, palette=sns.xkcd_palette(["blue", "orange", "red"]), ax=ax)
ax.plot("importance_cum", "feature", data=df_varimp.iloc[range(topn)], color="grey", marker="o")
ax.set_xlabel(r"importance / cumulative importance in % (-$\bullet$-)")
ax.set_title("Top{0: .0f} (of{1: .0f}) Feature Importances".format(topn, len(features)))
fig.tight_layout()
fig.savefig(plotloc + "variable_importance.pdf")


# --- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------
sns.barplot("importance_sumnormed", "feature", hue="fold",
            data=pd.concat([df_varimp_train.assign(fold="train"), df_varimp.assign(fold="test")], sort=False))


# ######################################################################################################################
# Partial Dependance
# ######################################################################################################################

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO


# ######################################################################################################################
# xgboost Explainer
# ######################################################################################################################

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
