# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

# ######################################################################################################################
# Initialize ||||----
# ######################################################################################################################

# --- Load libraries and functions -----------------------------------------------------------------------
# ETL
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import dill

# Plot
import matplotlib.pyplot as plt
# plt.ioff()
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
#from plotnine import *

# ML
from sklearn.model_selection import (GridSearchCV, KFold, PredefinedSplit, cross_validate, RepeatedKFold,
                                     learning_curve, cross_val_score)
from sklearn.metrics import (roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

#import os
#import sys
#sys.path.append(os.getcwd() + "\\code")
#from my_func import create_sparse_matrix


# Load result from exploration and run 0_init
dill.load_session("1_explore.pkl")
exec(open("./code/0_init.py").read())


# # Initialize parallel processing
# closeAllConnections()  # reset
# Sys.getenv("NUMBER_OF_PROCESSORS")
# cl = makeCluster(4)
# registerDoParallel(cl)
# customlibpaths =.libPaths()
# clusterExport(cl, "customlibpaths")
# clusterEvalQ(cl,.libPaths(customlibpaths))
# clusterEvalQ(cl, library(BoxCore))
# clusterEvalQ(cl, library(BoxAdvanced))
# # stopCluster(cl); closeAllConnections() #stop cluster

# Set metric for peformance comparison
metric = "roc_auc"


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid) ||||----
# ######################################################################################################################

# Sample data ----------------------------------------------------------------------------------------------------
# Sample from all data (take all but n_maxpersample at most)
# df_tune = df.groupby("target").apply(lambda x: x.sample(min(500, x.shape[0])))

# Undersample only training data
df_tune = pd.concat([df.loc[df.fold == "train", ].groupby("target").apply(lambda x: x.sample(min(500, x.shape[0]))),
                     df.loc[df.fold == "test", ]])
b_sample = df_tune.loc[df_tune.fold == "train", "target_num"].mean()
b_all = df.loc[df.fold == "train", "target_num"].mean()

df_tune["target"].describe()
print(b_sample, b_all)

from sklearn.model_selection import *

# Define some splits -------------------------------------------------------------------------------------------
split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_shuffle = ShuffleSplit(5, test_size = 0.2)
split_5fold = KFold(5, shuffle=False, random_state=42)


class MySplit:
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, folds=None):
        for i_train, i_test in KFold(n_splits=self.n_splits,
                                     shuffle=True,
                                     random_state=self.random_state).split(X, y):
            yield i_train[folds[i_train] == "train"], i_test[folds[i_test] == "test"]

    def get_n_splits(self, X, y, folds=None):
        return self.n_splits


split_mysplit = MySplit(random_state=42)
iter_split = split_mysplit.split(df_tune, df_tune.target_num, folds=df_tune.fold.values)
i_train, i_test = next(iter_split)
print(i_test)
df_tune.iloc[i_train, :].fold.describe()
df_tune.iloc[i_test, :].fold.describe()





# Fits -----------------------------------------------------------------------------------------------------------
# Lasso / Elastic Net
fit = GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                   [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                     "l1_ratio": [0, 0.2, 0.5, 0.8, 1]}],
                   # cv=ShuffleSplit(1, 0.2, random_state=999),
                   cv=split_index.split(),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4)\
    .fit(create_sparse_matrix(df_tune, cate=features_binned), df_tune["target_num"])
print(fit.best_params_)
pd.DataFrame.from_dict(fit.cv_results_)\
    .pivot_table(["mean_test_score"], index="param_alpha", columns="param_l1_ratio")\
    .plot(marker="o")
plt.close()
# -> keep l1_ratio=1 to have a full Lasso


# Random Forest
fit = GridSearchCV(RandomForestClassifier(warm_start=True),
                   [{"n_estimators": [10, 20, 50, 100, 200],
                     "max_features": [x for x in range(1, len(features), 3)]}],
                   cv=split_index,
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   # use_warm_start=["n_estimators"],
                   n_jobs=4)\
    .fit(create_sparse_matrix(df_tune, metr, cate), df_tune["target_num"])
print(fit.best_params_)
pd.DataFrame.from_dict(fit.cv_results_)\
    .pivot_table(["mean_test_score"], index="param_n_estimators", columns="param_max_features")\
    .plot(marker="o")
plt.close()
# -> keep around the recommended values: max_features = floor(sqrt(length(features)))

#
# # Boosted Trees
# fit = GridSearchCV(GradientBoostingClassifier(warm_start=True),
#                    [{"n_estimators": [x for x in range(100, 3100, 500)],
#                      "max_depth": [3, 6],
#                      "learning_rate": [0.01],
#                      "min_samples_leaf": [10]}],
#                    cv=split_index,
#                    refit=False,
#                    scoring=metric,
#                    return_train_score=True,
#                    n_jobs=4)\
#     .fit(create_sparse_matrix(df_tune, metr, cate), df_tune["target_num"])
# print(fit.best_params_)
# pd.DataFrame.from_dict(fit.cv_results_)\
#     .pivot_table(["mean_test_score"], index="param_n_estimators", columns="param_max_depth")\
#     .plot(marker="o")
# plt.close()

fit = GridSearchCV(xgb.XGBClassifier(),
                   [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01, 0.02],
                     "max_depth": [2, 3], "min_child_weight": [5, 10]}],
                   cv=split_index,
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4) \
    .fit(create_sparse_matrix(df_tune, metr, cate), df_tune["target_num"])
print(fit.best_params_)
# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10

df_fitres = pd.DataFrame.from_dict(fit.cv_results_)
df_fitres["param_min_child_weight__learning_rate"] = (df_fitres.param_min_child_weight.astype("str") + "_" +
                                                      df_fitres.param_learning_rate.astype("str"))
sns.catplot(kind="point",
            data=df_fitres,
            x="param_n_estimators", y="mean_test_score", hue="param_min_child_weight__learning_rate",
            col="param_max_depth",
            palette=["C0", "C0", "C1", "C1"], markers=["o", "x", "o", "x"], linestyles=["-", ":", "-", ":"],
            legend_out=False)
plt.close()
# factors = ["param_min_child_weight", "param_learning_rate", "param_max_depth"]
# df_fitres[factors] = df_fitres[factors].astype("str")
# (ggplot(df_fitres, aes(x="param_n_estimators",
#                        y="mean_test_score",
#                        colour="param_min_child_weight"))
#  + geom_line(aes(linetype="param_learning_rate"))
#  + geom_point(aes(shape="param_learning_rate"))
#  + facet_grid(". ~ param_max_depth"))

fit = GridSearchCV(lgbm.LGBMClassifier(),
                   [{"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                     "num_leaves": [8, 32], "min_child_samples": [10]}],
                   cv=split_index,
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4) \
    .fit(df_tune[features_lgbm], df_tune["target_num"],
         categorical_feature=[x for x in features_lgbm.tolist() if "_ENCODED" in x]
         )
print(fit.best_params_)
pd.DataFrame.from_dict(fit.cv_results_) \
    .pivot_table(["mean_test_score"], index="param_n_estimators", columns="param_num_leaves") \
    .plot(marker="o")
plt.close()


# Score 

fit = RandomForestClassifier(n_estimators=50, max_features=3)\
    .fit(create_sparse_matrix(df_tune, metr, cate), df_tune.target_num)
yhat = fit.predict_proba(create_sparse_matrix(df_tune, metr, cate))[:, 1]
roc_auc_score(df_tune.target_num, yhat)
fpr, tpr, cutoff = roc_curve(df_tune.target_num, yhat)
cross_val_score(fit,
                create_sparse_matrix(df_tune, metr, cate), df_tune.target_num,
                cv=5, scoring=metric, n_jobs=5)





# ######################################################################################################################
# Evaluate generalization gap
# ######################################################################################################################

# Sample data (usually undersample training data)
df_gengap = df_tune

# Tune grid to loop over
param_grid = [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
               "max_depth": [3, 6], "min_child_weight": [5, 10],
               "colsample_bytree": [0.7], "subsample": [0.7],
               "gamma": [10]}]

# Calc generalization gap
fit = GridSearchCV(xgb.XGBClassifier(),
                   param_grid,
                   cv=PredefinedSplit(df_gengap["fold"].map({"train": -1, "test": 0}).values),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4) \
    .fit(create_sparse_matrix(df_gengap, metr, cate), df_gengap["target_num"])

# Plot generalization gap
df_fitres = pd.DataFrame.from_dict(fit.cv_results_)\
    .rename(columns={"mean_test_score": "score_test",
                     "mean_train_score": "score_train"})\
    .assign(train_test_score_diff=lambda x: x.score_train - x.score_test)\
    .reset_index()
groupcols = ["param_max_depth","param_min_child_weight"]
df_fitres[groupcols] = df_fitres[groupcols].apply(lambda x: "_" + x.astype(str))
sns.catplot(kind="point",
            data=df_fitres,
            x="param_n_estimators", y="train_test_score_diff", hue="param_max_depth",
            col="param_min_child_weight",
            legend_out=False)
sns.lineplot(data=df_fitres,
             x="param_n_estimators", y="train_test_score_diff", hue="param_min_child_weight",
             size="param_min_child_weight",
             style = "param_max_depth"
             )

df_plot = pd.wide_to_long(df_fitres, 'score', i='index', j='fold', sep='_', suffix="\\w+").reset_index()
sns.catplot(kind="point",
            data=df_plot,
            x="param_n_estimators", y="score", hue="fold",
            col="param_min_child_weight", row="param_max_depth",
            legend_out=False)
sns.lineplot(data=df_plot,
             x="param_n_estimators", y="score", hue="fold",
             size="param_min_child_weight", style="param_max_depth",
             markers=True)


# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_sim = df_tune

df_sim_result = pd.DataFrame()


# --- Run methods ------------------------------------------------------------------------------
# Elastic Net
cvresults = cross_validate(
      estimator = GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                               [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                                 "l1_ratio": [1]}],
                               cv=ShuffleSplit(1, 0.2, random_state=999),
                               refit=True,
                               scoring=metric,
                               return_train_score=False,
                               n_jobs=4),
      X=create_sparse_matrix(df_sim, cate=features_binned),
      y=df_sim["target_num"],
      cv=MySplit(5, random_state=42).split(df_sim, df_sim.target_num, df_sim.fold),
      return_train_score=False,
      n_jobs=4)
df_sim_result = df_sim_result.append(pd.DataFrame.from_dict(cvresults).reset_index().assign(model = "ElasticNet"),
                                     ignore_index=True)

# Xgboost
cvresults = cross_validate(
      estimator=GridSearchCV(xgb.XGBClassifier(),
                             [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                               "max_depth": [6], "min_child_weight": [10]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),
                             refit=True,
                             scoring=metric,
                             return_train_score=False,
                             n_jobs=4),
      X=create_sparse_matrix(df_sim, metr, cate),
      y=df_tune["target_num"],
      cv=RepeatedKFold(5, 2, random_state=42),
      return_train_score=False,
      n_jobs=4)
df_sim_result = df_sim_result.append(pd.DataFrame.from_dict(cvresults).reset_index().assign(model = "XGBoost"),
                                     ignore_index=True)


# --- Plot ------------------------------------------------------------------------------
sns.boxplot(data=df_sim_result, x="model", y="test_score")
sns.lineplot(data=df_sim_result, x="model", y="test_score", hue="index", linewidth=0.5, linestyle=":")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################


