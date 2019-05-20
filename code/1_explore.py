# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

# ######################################################################################################################
#  Libraries + Parallel Processing Start
# ######################################################################################################################

# Load libraries and functions
# exec(open("./code/0_init.py").read())
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.stats import chi2_contingency
import dill
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# from plotnine import *
from sklearn.model_selection import (GridSearchCV, ShuffleSplit, cross_validate, RepeatedKFold, learning_curve,
                                     cross_val_score)
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

import pdb # pdb.set_trace() # quit with "q", next line with "n", continue with "c"

# import os
# os.getcwd()
# os.chdir("C:/My/Projekte/Python_PredMod")
exec(open("./code/0_init.py").read())


# ######################################################################################################################
# Initialize
# ######################################################################################################################

# Adapt some default parameter differing per target types (NULL results in default value usage)
# cutoff_switch = switch(TARGET_TYPE, "CLASS" = 0.1, "REGR" = 0.8, "MULTICLASS" = 0.8)  # adapt
# ylim_switch = switch(TARGET_TYPE, "CLASS" = NULL, "REGR" = c(0, 250e3), "MULTICLASS" = NULL)  # adapt REGR opt
cutoff_corr = 0.5
cutoff_varimp = 0.52


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data ------------------------------------------------------------------------------------------------------
df_orig = pd.read_csv(dataloc + "titanic.csv")
df_orig.describe()
df_orig.describe(include=["object"])

"""
# Check some stuff
df_values = create_values_df(df_orig, 10)
print(df_values)
df_orig["survived"].value_counts() / df_orig.shape[0]
"""

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) -------------------------------------------------------------------------------
df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)

# Check
print(np.setdiff1d(df.columns.values, df_meta["variable"]))
print(np.setdiff1d(df_meta.loc[df_meta["category"] == "orig", "variable"], df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready", "derive"])]


# --- Feature engineering -----------------------------------------------------------------------------------------
# df$deck = as.factor(str_sub(df$cabin, 1, 1))
df["deck"] = df["cabin"].str[:1]
df["familysize"] = df["sibsp"] + df["parch"] + 1
df["fare_pp"] = df.groupby("ticket")["fare"].transform("mean")
df[["deck", "familysize", "fare_pp"]].describe(include="all")

# Check
print(np.setdiff1d(df_meta["variable"], df.columns.values))


# --- Define target and train/test-fold ----------------------------------------------------------------------------
# Target
df["target"] = df["survived"]
df["target"].describe()

# Train/Test fold: usually split by time
df["fold"] = "train"
df.loc[df.sample(frac=0.3, random_state=123).index, "fold"] = "test"
print(df.fold.value_counts())
df["fold_num"] = df["fold"].map({"train": 0, "test": 1})

# Define the id
df["id"] = np.arange(len(df)) + 1


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------
metr = df_meta_sub.loc[df_meta_sub.type == "metr", "variable"].values
df[metr] = df[metr].apply(pd.to_numeric)
df[metr].describe()


# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
metr_binned = metr + "_BINNED_"
df[metr_binned] = df[metr].apply(lambda x: pd.qcut(x, 10).astype("str"))

# Convert missings to own level ("(Missing)")
df[metr_binned] = df[metr_binned].fillna("(Missing)")
print(create_values_df(df[metr_binned], 11))

# Remove binned variables with just 1 bin
onebin = metr_binned[df[metr_binned].nunique() == 1]
metr_binned = np.setdiff1d(metr_binned, onebin)


# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------
# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
misspct.sort_values(ascending=False)  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
print(remove)
metr = np.setdiff1d(metr, remove)  # adapt metadata
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED_")  # keep "binned" version in sync

# Check for outliers and skewness
plot_distr(df, metr, ncol=1, nrow=2, w=8, h=6)
plt.close(fig="all")

# Winsorize
df[metr] = df[metr].apply(lambda x: winsorize(x, (0.01, 0.01)))  # hint: plot again before deciding for log-trafo

# Log-Transform
tolog = np.array(["fare"], dtype="object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - max(0, np.min(x)) + 1))
np.place(metr, np.isin(metr, tolog), tolog + "_LOG_")  # adapt metadata (keep order)


# --- Final variable information ------------------------------------------------------------------------------------
# Univariate variable importance
varimp_metr = calc_varimp(df, metr)
print(varimp_metr)

# Plot 
plot_distr(df, metr, varimp=varimp_metr, ncol=2, nrow=2, w=18, h=12, pdf=plotloc + "distr_metr.pdf")


# --- Removing variables -------------------------------------------------------------------------------------------
# Remove leakage features
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED")  # keep "binned" version in sync

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df, metr, cutoff=cutoff_corr, pdf=plotloc + "corr_metr.pdf")
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED")  # keep "binned" version in sync


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_fold = calc_varimp(df, metr, "fold_num")

# Plot: only variables with with highest importance
metr_toprint = varimp_metr_fold[varimp_metr_fold > cutoff_varimp].index.values
plot_distr(df, metr_toprint, "fold_num", varimp=varimp_metr_fold, ncol=2, nrow=2, w=18, h=12,
           pdf=plotloc + "distr_metr_final_folddependency.pdf")
plt.close(plt.gcf())


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------

miss = metr[df[metr].isnull().any().values]
# alternative: [x for x in metr if df[x].isnull().any()]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
df[miss] = df[miss].fillna(df[miss].median())
# df[miss] = df[miss].apply(lambda x: np.where(x.isnull(), np.random.choice(x[x.notnull()], len(x.isnull())), x))
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------
# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["nomi", "ordi"]), "variable"].values
df[cate] = df[cate].astype("str")
df[cate].describe()

# Merge categorical variable (keep order)
cate = np.append(cate, ["MISS_" + miss])


# --- Handling factor values ----------------------------------------------------------------------------------------
# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 10
levinfo = df[cate].apply(lambda x: x.unique().size).sort_values(ascending=False)  # number of levels
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = np.setdiff1d(toomany, ["xxx", "xxx"])  # set exception for important variables
df[cate + "_ENCODED"] = df[cate]

# Convert toomany features: lump levels and map missings to own level
df[toomany] = df[toomany].apply(lambda x:
                                x.replace(np.setdiff1d(x.value_counts()[topn_toomany:].index.values, "(Missing)"),
                                          "_OTHER_"))

# Create encoded features (for tree based models), i.e. numeric representation
df[cate + "_ENCODED"] = df[cate + "_ENCODED"].apply(
    lambda x: x.replace(x.value_counts().index.values.tolist(),
                        (np.arange(len(x.value_counts())) + 1).tolist()))

# Univariate variable importance
varimp_cate = calc_varimp(df, cate)
print(varimp_cate)

# Check
plot_distr(df, cate, varimp=varimp_cate, ncol=2, nrow=2, w=8, h=6, pdf=plotloc + "cate.pdf")


# --- Removing variables ---------------------------------------------------------------------------------------------
# Remove leakage variables
cate = np.setdiff1d(cate, ["boat"])
toomany = np.setdiff1d(toomany, ["boat"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df, cate, cutoff=cutoff_corr, pdf=plotloc + "corr_cate.pdf")


# --- Time/fold depedency --------------------------------------------------------------------------------------------
# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_fold = calc_varimp(df, cate, "fold_num")

# Plot: only variables with with highest importance
cate_toprint = varimp_cate_fold[varimp_cate_fold > cutoff_varimp].index.values
plot_distr(df, cate_toprint, "fold_num", varimp=varimp_cate_fold, ncol=2, nrow=2, w=12, h=8,
           pdf=plotloc + "distr_cate_final_folddependency.pdf")
plt.close(plt.gcf())


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features ----------------------------------------------------------------------------------------
features_notree = np.append(metr, cate)
features_lgbm = np.append(metr, cate + "_ENCODED")
features = np.append(features_notree, toomany + "_ENCODED")
features_binned = np.append(metr_binned, np.setdiff1d(cate, "MISS_" + miss))  # do not need indicators for binned

# Check
np.setdiff1d(features_notree, df.columns.values.tolist())
np.setdiff1d(features, df.columns.values.tolist())
np.setdiff1d(features_binned, df.columns.values.tolist())


# --- Save image ------------------------------------------------------------------------------------------------------
del df_orig
dill.dump_session("1_explore.pkl")
