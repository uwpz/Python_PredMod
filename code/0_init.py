# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

# ######################################################################################################################
# Libraries ----
# ######################################################################################################################

# import numpy as np
# import pandas as pd
# from scipy.stats.mstats import winsorize
# import dill
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# # from plotnine import *
# from sklearn.model_selection import (GridSearchCV, ShuffleSplit, PredefinedSplit, cross_validate, RepeatedKFold, learning_curve,
#                                      cross_val_score)
# from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import ElasticNet
# import xgboost as xgb
# import lightgbm as lgbm


# ######################################################################################################################-
# Parameters ----
# ######################################################################################################################-

dataloc = "./data/"
plotloc = "./output/"

sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)


# ######################################################################################################################
# My Functions
# ######################################################################################################################

# def setdiff(a, b):
#     return [x for x in a if x not in set(b)]

# def union(a, b):
#     return a + [x for x in b if x not in set(a)]

def create_values_df(df_, topn):
    return pd.concat([df_[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_c"})
                      for catname in df_.select_dtypes(["object"]).columns.values], axis=1)


def create_sparse_matrix(data, metr=None, cate=None):
    if metr is not None:
        m_metr = data[metr].to_sparse().to_coo()
    else:
        m_metr = None
    if cate is not None:
        if len(cate) == 1:
            m_cate = OneHotEncoder().fit_transform(data[cate].reshape(-1, 1))
        else:
            m_cate = OneHotEncoder().fit_transform(data[cate])
    else:
        m_cate = None
    return hstack([m_metr, m_cate])