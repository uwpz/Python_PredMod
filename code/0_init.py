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


def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def plot_distr(df_data, features, target_name="target", varimp=None,
               ncol=1, nrow=1, w=8, h=6, pdf=None):
    # df_data = df; features = metr_toprint; target_name = "fold_num";  varimp=varimp_metr_fold;
    # ncol=2; nrow=2; pdf=None; w=8; h=6

    # Help variables
    n_ppp = ncol * nrow  # plots per page
    n_pages = len(features) // n_ppp + 1  # number of pages

    # Open pdf
    if pdf is not None:
        pdf_pages = PdfPages(pdf)

    # Plot
    # l_fig = list()
    for page in range(n_pages):
        # page = 0
        fig, ax = plt.subplots(ncol, nrow)
        for i in range(n_ppp):
            # i = 1
            ax_act = ax.flat[i]
            if page * n_ppp + i <= max(range(len(features))):
                feature_act = features[page * n_ppp + i]

                # Categorical feature
                if df_data[feature_act].dtype == "object":
                    # Prepare data
                    df_plot = pd.DataFrame({"h": df_data.groupby(feature_act)[target_name].mean(),
                                            "w": df_data.groupby(feature_act).size()}).reset_index()
                    df_plot.w = df_plot.w/max(df_plot.w)
                    df_plot["new_w"] = np.where(df_plot["w"].values < 0.2, 0.2, df_plot["w"])

                    # Target barplot
                    # sns.barplot(df_tmp.h, df_tmp[cate[page * ppp + i]], orient="h", color="coral", ax=axact)
                    ax_act.barh(df_plot[feature_act], df_plot.h, height=df_plot.new_w, edgecolor="black")
                    ax_act.set_xlabel("Proportion Target = Y")
                    if varimp is not None:
                        ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                    ax_act.axvline(np.mean(df_data[target_name]), ls="dotted", color="black")

                    # Inner distribution plot
                    xlim = ax_act.get_xlim()
                    ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
                    inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
                    inset_ax.set_axis_off()
                    ax_act.get_shared_y_axes().join(ax_act, inset_ax)
                    ax_act.axvline(0, color="black")
                    inset_ax.barh(df_plot[feature_act], df_plot.w, color="grey")


                # Metric feature
                else:
                    sns.distplot(df_data.loc[df_data[target_name] == 1, feature_act].dropna(), color="red", label="1", ax=ax_act)
                    sns.distplot(df_data.loc[df_data[target_name] == 0, feature_act].dropna(), color="blue", label="0", ax=ax_act)
                    # sns.FacetGrid(df_data, hue=target_name, palette=["red","blue"])\
                    #     .map(sns.distplot, metr[i])\
                    #     .add_legend() # does not work for multiple axes
                    if varimp is not None:
                        ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                    ax_act.set_ylabel("density")
                    ax_act.set_xlabel(feature_act + "(NA: " +
                                      str(df_data[feature_act].isnull().mean().round(3) * 100) +
                                      "%)")

                    # Boxplot
                    ylim = ax_act.get_ylim()
                    ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
                    inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
                    inset_ax.set_axis_off()
                    ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                    i_bool = df_data[feature_act].notnull()
                    sns.boxplot(x=df_data.loc[i_bool, feature_act],
                                y=df_data.loc[i_bool, target_name].astype("category"),
                                palette=["blue", "red"],
                                ax=inset_ax)
                    ax_act.legend(title=target_name, loc="best")
                # plt.show()
            else:
                ax_act.axis("off")  # Remove left-over plots
        # plt.subplots_adjust(wspace=1)
        fig.set_size_inches(w=w, h=h)
        fig.tight_layout()
        # l_fig.append(fig)
        if pdf is not None:
            pdf_pages.savefig(fig)
            plt.close(fig)
    if pdf is not None:
        pdf_pages.close()


# Univariate variable importance
def calc_varimp(df_data, features, target_name="target"):
    # df_data=df; features=metr; target_name="fold"
    varimp = pd.Series()
    for feature_act in features:
        # feature_act=metr[0]

        if df_data[feature_act].dtype == "object":
            varimp_act = {feature_act: (roc_auc_score(y_true=df_data[target_name].values,
                                                      y_score=df_data[[feature_act, target_name]]
                                                      .groupby(feature_act)[target_name]
                                                      .transform("mean").values)
                                        .round(3))}
        else:
            varimp_act = {feature_act: (roc_auc_score(y_true=df_data[target_name].values,
                                                      y_score=df_data[[target_name]]
                                                      .assign(dummy=pd.qcut(df_data[feature_act], 10).astype("object")
                                                              .fillna("(Missing)"))
                                                      .groupby("dummy")[target_name]
                                                      .transform("mean").values)
                                        .round(3))}
        varimp = varimp.append(pd.Series(varimp_act))
    varimp.sort_values(ascending=False, inplace=True)
    return varimp


# Plot correlation
def plot_corr(df_data, features, cutoff=0, w=8, h=6, pdf=None):
    # df_data = df; features = cate; cutoff = 0.1; w=8; h=6; pdf="blub.pdf"

    metr = features[df_data[features].dtypes != "object"]
    cate = features[df_data[features].dtypes == "object"]

    if len(metr) and len(cate):
        raise Exception('Mixed dtypes')
        return

    if len(cate):
        df_corr = pd.DataFrame(np.ones([len(cate),len(cate)]), index = cate, columns = cate)
        for i in range(len(cate)):
            print("cate=",cate[i])
            for j in range(i+1, len(cate)):
                #i=1; j=2
                tmp = pd.crosstab(df_data[features[i]], df_data[features[j]])
                n = np.sum(tmp.values)
                M = min(tmp.shape)
                chi2 = chi2_contingency(tmp)[0]
                df_corr.iloc[i,j] = np.sqrt(chi2 / (n + chi2)) * np.sqrt(M / (M-1))
                df_corr.iloc[j,i] = df_corr.iloc[i,j]

    if len(metr):
        df_corr = abs(df_data[metr].corr(method="spearman"))

    # Cut off
    np.fill_diagonal(df_corr.values, 0)
    i_bool = (df_corr.max(axis=1) > cutoff).values
    df_corr = df_corr.loc[i_bool, i_bool]
    np.fill_diagonal(df_corr.values, 1)

    # Plot
    fig, ax = plt.subplots(1, 1)
    ax_act = ax
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Blues", ax=ax_act)
    ax_act.set_yticklabels(labels=ax_act.get_yticklabels(), rotation=0)
    ax_act.set_xticklabels(labels=ax_act.get_xticklabels(), rotation=90)
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        plt.close(fig)

