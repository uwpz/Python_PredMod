
# ######################################################################################################################
# Libraries
# ######################################################################################################################

# Data
import numpy as np
import pandas as pd

# Plot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns

# ETL
import scipy as sp
from scipy.stats import chi2_contingency
from scipy.sparse import hstack
from scipy.cluster.hierarchy import ward, fcluster

# ML
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, SGDClassifier, SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree #, export_graphviz
import xgboost as xgb
from sklearn.model_selection import (KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_validate, ShuffleSplit, PredefinedSplit,
                                     learning_curve, cross_val_predict)
# from sklearn.externals.six import StringIO
# from glmnet_python import glmnet, glmnetPredict

# Util
from collections import defaultdict
import os
from os import getcwd
import pdb  # pdb.set_trace()  #quit with "q", next line with "n", continue with "c"
from joblib import Parallel, delayed
#from dill import (load_session, dump_session)
import pickle
#from IPython.display import Image
#import pydotplus
#import xlwt

# Silent plotting (Overwrite to get default: plt.ion();  matplotlib.use('TkAgg'))
plt.ioff(); matplotlib.use('Agg')
#plt.ion();  matplotlib.use('TkAgg')

# ######################################################################################################################
# Parameters
# ######################################################################################################################

# Locations
dataloc = "./data/"
plotloc = "./output/"

# Util
sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

# Other
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
colors = pd.Series(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
colors = colors.iloc[np.setdiff1d(np.arange(len(colors)),[6,7,8,9,12,13,14,15,16,17,26])]
#sel = np.arange(50);  plt.bar(sel.astype("str"), 1, color=colors[sel])


# ######################################################################################################################
# My Functions and Classes
# ######################################################################################################################



# --- Functions ----------------------------------------------------------------------------------------

def setdiff(a, b):
    return np.setdiff1d(a, b, True)


# Overview of values
def create_values_df(df, topn):
    return pd.concat([df[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_#"})
                      for catname in df.dtypes.index.values[df.dtypes == "object"]], axis=1)


def char_bins(series, n_bins=10, prefix="q"):
    bins = pd.qcut(series, n_bins, duplicates="drop")
    bins.cat.categories = [prefix + str(i).zfill(2)  # + ":" + bins.cat.categories.astype("str")[i-1]
                           for i in 1 + np.arange(len(bins.cat.categories))]
    bins = bins.astype("str").replace("nan", np.nan)
    return bins


def spear(y_true, y_pred):
    spear = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="spearman").values[0, 1]
    return spear


def rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    return rmse


def auc(y_true, y_pred):
    #pdb.set_trace()
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
    return auc


def acc(y_true, y_pred):
    # pdb.set_trace()
    acc = accuracy_score(y_true, y_pred)
    return acc


# Scoring metrics
scoring = {"CLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                     "acc": make_scorer(acc, greater_is_better=True)},
           "MULTICLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                          "acc": make_scorer(acc, greater_is_better=True)},
           "REGR": {"spear": make_scorer(spear, greater_is_better=True),
                    "rmse": make_scorer(rmse, greater_is_better=False)}}


# Rescale predictions (e.g. to rewind undersampling)
def scale_predictions(yhat, b_sample=None, b_all=None):
    if b_sample is None:
        yhat_rescaled = yhat
    else:
        # tmp = yhat * np.array([1 - b_all, b_all]) / np.array([1 - b_sample, b_sample])
        tmp = (yhat * b_all) / b_sample
        yhat_rescaled = (tmp.T / tmp.sum(axis=1)).T  # transposing is needed for casting
    return yhat_rescaled


# Show closed figure again
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    

# Univariate variable importance
def calc_imp(df, features, target="target", target_type="CLASS"):
    #df=df; features=metr; target="target"; target_type="MULTICLASS"
    varimp = pd.Series()
    for feature_act in features:
        # feature_act=cate[0]
        if target_type == "CLASS":
            try:
                y_true = df[target].values
                if df[feature_act].dtype == "object":
                    dummy = df[feature_act]
                else:
                    # Create bins from metric variable
                    dummy = pd.qcut(df[feature_act], 10, duplicates="drop").astype("object").fillna("(Missing)")
                y_score = df[[target]].groupby(dummy).transform("mean").values
                varimp_act = {feature_act: roc_auc_score(y_true, y_score).round(3)}
            except:
                varimp_act = {feature_act: 0.5}

        if target_type == "MULTICLASS":
            # Similar to "CLASS" but now y_true and y_pred are matrices
            try:
                y_true = df[target].str.get_dummies()
                if df[feature_act].dtype == "object":
                    dummy = df[feature_act]
                else:
                    dummy = pd.qcut(df[feature_act], 10, duplicates="drop").astype("object").fillna("(Missing)")
                tmp = pd.crosstab(dummy, df[target])
                y_score = (pd.DataFrame(dummy).reset_index()
                           .merge(tmp.div(tmp.sum(axis=1), axis=0).reset_index(), how="inner")
                           .sort_values("index")
                           .reset_index(drop=True)[y_true.columns.values])
                varimp_act = {feature_act: roc_auc_score(y_true, y_score).round(3)}
            except:
                varimp_act = {feature_act: 0.5}

        if target_type == "REGR":
            y_true = df[target]
            if df[feature_act].dtype == "object":
                y_score = df.groupby(feature_act)[target].transform("mean")
            else:
                y_score = df[feature_act]
            varimp_act = {feature_act: (abs(pd.DataFrame({"y_true": y_true, "y_score": y_score})
                                            .corr(method="spearman")
                                            .values[0, 1]).round(3))}

        varimp = varimp.append(pd.Series(varimp_act))
    varimp.sort_values(ascending=False, inplace=True)
    return varimp
# Plot distribution regarding target
def plot_distr(df, features, target="target", target_type="CLASS",
               varimp=None, color=["blue", "red"], ylim=None, regplot=True, min_width=0,
               nrow=1, ncol=1, w=8, h=6, pdf=None):
    # df = df; features = cate; target = "target"; target_type="MULTICLASS"; color=threecol; varimp=None; min_width=0
    # ylim = ylim; ncol=3; nrow=3; pdf=None; w=8; h=6

    # Help variables
    n_ppp = ncol * nrow  # plots per page

    # Open pdf
    if pdf is not None:
        pdf_pages = PdfPages(pdf)
    else:
        pdf_pages = None

    # Plot (loop over features)
    for i, feature_act in enumerate(features):
        #i=0; feature_act=features[i]

        # Start new subplot on new page
        if i % n_ppp == 0:
            fig, ax = plt.subplots(nrow, ncol)
            fig.set_size_inches(w=w, h=h)
            i_ax = 0

        # Catch single plot case
        if n_ppp == 1:
            ax_act = ax
        else:
            ax_act = ax.flat[i_ax]

        # Distinguish first by target_type ...
        if target_type in ["CLASS", "MULTICLASS"]:

            # ... then by feature type (metric features)
            if df[feature_act].dtype != "object":

                # Main distribution plot (overlayed)
                members = np.sort(df[target].unique())
                for m, member in enumerate(members):
                    sns.distplot(df.loc[df[target] == member, feature_act].dropna(), color=color[m],
                                 bins=20, label=member, ax=ax_act)
                if varimp is not None:
                    ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
                else:
                    ax_act.set_title(feature_act)
                ax_act.set_ylabel("density")
                ax_act.set_xlabel(feature_act + " (NA: " +
                                  str((df[feature_act].isnull().mean() * 100).round(1)) +
                                  "%)")
                ax_act.legend(title=target, loc="best")

                # Inner Boxplot
                ylim = ax_act.get_ylim()
                ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
                inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
                inset_ax.set_axis_off()
                ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                i_bool = df[feature_act].notnull()
                sns.boxplot(x=df.loc[i_bool, feature_act],
                            y=df.loc[i_bool, target].astype("category"),
                            showmeans=True,
                            meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"},
                            palette=color,
                            ax=inset_ax)

            # Categorical feature
            else:
                #pdb.set_trace()
                # Prepare data
                # df_plot = pd.DataFrame({"h": df.groupby(feature_act)[target].mean(),
                #                         "w": df.groupby(feature_act).size()}).reset_index()
                # df_plot = df.groupby(feature_act)[target].agg(h="mean", w="size").reset_index()
                # df_plot["pct"] = 100 * df_plot["w"] / len(df)
                # df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
                # df_plot[feature_act + "_new"] = df_plot[feature_act] + " (" + \
                #                                 (df_plot["pct"]).round(1).astype(str) + "%)"
                # df_plot["new_w"] = np.where(df_plot["w"].values < min_width, min_width, df_plot["w"])


                df_hlp = pd.crosstab(df[feature_act], df[target])
                df_plot = df_hlp.div(df_hlp.sum(axis=1), axis=0)
                df_plot["w"] = df_hlp.sum(axis=1)
                df_plot = df_plot.reset_index()
                df_plot["pct"] = 100 * df_plot["w"] / len(df)
                df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
                df_plot[feature_act + "_new"] = df_plot[feature_act] + " (" + \
                                                (df_plot["pct"]).round(1).astype(str) + "%)"
                df_plot["new_w"] = np.where(df_plot["w"].values < min_width, min_width, df_plot["w"])

                #pdb.set_trace()
                # Main barplot
                if target_type == "MULTICLASS":
                    offset = np.zeros(len(df_plot))
                    for m, member in enumerate(np.sort(df[target].unique())):
                        ax_act.barh(df_plot[feature_act + "_new"], df_plot[member], height=df_plot.new_w, left=offset,
                                    color=color[m], label=member, edgecolor="black", alpha=0.5, linewidth=1)
                        offset = offset + df_plot[member].values
                    ax_act.legend(title=target, loc="center left", bbox_to_anchor=(1, 0.5))

                else:
                    ax_act.barh(df_plot[feature_act + "_new"], df_plot[1], height=df_plot.new_w,
                                color=color[1], edgecolor="black", alpha=0.5, linewidth=1)
                ax_act.set_xlabel("mean(" + target + ")")
                #ax_act.set_yticklabels(df_plot[feature_act + "_new"].values)
                #ax_act.set_yticklabels(df_plot[feature_act].values)
                if varimp is not None:
                    ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
                else:
                    ax_act.set_title(feature_act)
                if target_type == "CLASS":
                    ax_act.axvline(np.mean(df[target]), ls="dotted", color="black")  # priori line

                # Inner barplot
                xlim = ax_act.get_xlim()
                ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
                inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
                inset_ax.set_axis_off()
                ax_act.axvline(0, color="black")  # separation line
                # inset_ax.set_yticklabels(df_plot[feature_act + "_new"].values)
                ax_act.get_shared_y_axes().join(ax_act, inset_ax)
                inset_ax.barh(df_plot[feature_act + "_new"], df_plot.w,
                              color="lightgrey", edgecolor="black", linewidth=1)

        if target_type == "REGR":

            # Metric feature
            if df[feature_act].dtype != "object":

                # Main Heatmap

                # Calc scale
                if ylim is not None:
                    ax_act.set_ylim(ylim)
                    tmp_scale = (ylim[1] - ylim[0]) / (np.max(df[target]) - np.min(df[target]))
                else:
                    tmp_scale = 1

                # Calc colormap
                tmp_cmap = mcolors.LinearSegmentedColormap.from_list("gr_bl_yl_rd",
                                                                     [(0.5, 0.5, 0.5, 0), "blue", "yellow",
                                                                      "red"])
                # Hexbin plot
                ax_act.set_facecolor('0.98')
                p = ax_act.hexbin(df[feature_act], df[target],
                                  gridsize=(int(50 * tmp_scale), 50),
                                  cmap=tmp_cmap)
                plt.colorbar(p, ax=ax_act)
                if varimp is not None:
                    ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
                else:
                    ax_act.set_title(feature_act)
                ax_act.set_ylabel(target)
                ax_act.set_xlabel(feature_act + " (NA: " +
                                  str(df[feature_act].isnull().mean().round(3) * 100) +
                                  "%)")
                ylim = ax_act.get_ylim()
                # ax_act.grid(False)
                ax_act.axhline(color="grey")

                # Add lowess regression line?
                if regplot:
                    sns.regplot(feature_act, target, df, lowess=True, scatter=False, color="black", ax=ax_act)

                # Inner Histogram
                ax_act.set_ylim(ylim[0] - 0.4 * (ylim[1] - ylim[0]))
                inset_ax = ax_act.inset_axes([0, 0.07, 1, 0.2])
                inset_ax.set_axis_off()
                inset_ax.get_yaxis().set_visible(False)
                ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                i_bool = df[feature_act].notnull()
                sns.distplot(df[feature_act].dropna(), bins=20, color="black", ax=inset_ax)
                inset_ax.set_xlabel("")

                # Inner-inner Boxplot
                inset_ax = ax_act.inset_axes([0, 0.01, 1, 0.05])
                # ylim_inner = inset_ax.get_ylim()
                # inset_ax.set_ylim(ylim_inner[0] - 0.35 * (ylim_inner[1] - ylim_inner[0]))
                inset_ax.set_axis_off()
                inset_ax.get_shared_x_axes().join(ax_act, inset_ax)
                sns.boxplot(x=df.loc[i_bool, feature_act], palette=["grey"], ax=inset_ax)
                # inset_ax.set_xlabel("")
                inset_ax.set_xlabel(feature_act + " (NA: " +
                                    str(df[feature_act].isnull().mean().round(3) * 100) +
                                    "%)")

            # Categorical feature
            else:
                # Prepare data (Same as for CLASS target)
                df_plot = pd.DataFrame({"h": df.groupby(feature_act)[target].mean(),
                                        "w": df.groupby(feature_act).size()}).reset_index()
                df_plot["pct"] = 100 * df_plot["w"] / len(df)
                df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
                df_plot[feature_act + "_new"] = df_plot[feature_act] + " (" + \
                                                (df_plot["pct"]).round(1).astype(str) + "%)"
                df_plot["new_w"] = np.where(df_plot["w"].values < min_width, min_width, df_plot["w"])

                # Main grouped boxplot
                if ylim is not None:
                    ax_act.set_xlim(ylim)
                bp = df[[feature_act, target]].boxplot(target, feature_act, vert=False,
                                                       widths=df_plot.w.values,
                                                       showmeans=True,
                                                       meanprops=dict(marker="x",
                                                                      markeredgecolor="black"),
                                                       flierprops=dict(marker="."),
                                                       return_type='dict',
                                                       ax=ax_act)
                [[item.set_color('black') for item in bp[target][key]] for key in bp[target].keys()]
                fig.suptitle("")
                ax_act.set_xlabel(target)
                ax_act.set_yticklabels(df_plot[feature_act + "_new"].values)
                if varimp is not None:
                    ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
                else:
                    ax_act.set_title(feature_act)
                ax_act.axvline(np.mean(df[target]), ls="dotted", color="black")

                # Inner barplot
                xlim = ax_act.get_xlim()
                ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
                inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
                inset_ax.set_axis_off()
                if ylim is not None:
                    ax_act.axvline(ylim[0], color="black")
                df_plot.plot.barh(y="w", x=feature_act,
                                  color="lightgrey", ax=inset_ax, edgecolor="black", linewidth=1, legend=False)
                # inset_ax.barh(df_plot[feature_act], df_plot.w, color="lightgrey", edgecolor="black",
                #              linewidth=1)

        i_ax += 1

        # Write figures
        if i_ax == n_ppp or i == len(features)-1:
            if i == len(features)-1:
                for k in range(i_ax, nrow*ncol):
                    # Remove unused axes
                    ax.flat[k].axis("off")
            fig.tight_layout()
            pdf_pages.savefig(fig)

    # Close pdf
    if pdf is not None:
        pdf_pages.close()


# Plot correlation
def plot_corr(df, features, type="contingency", cutoff=0, n_cluster=5, w=8, h=6, pdf=None):
    # df = df; features = cate; cutoff = 0; n_cluster=3; w=8; h=6; pdf="blub.pdf"

    # Check for mixed types
    metr = features[df[features].dtypes != "object"]
    cate = features[df[features].dtypes == "object"]
    if len(metr) and len(cate):
        raise Exception('Mixed dtypes')
        # return

    # All categorical variables
    if len(cate):
        # Intialize matrix with zeros
        df_corr = pd.DataFrame(np.zeros([len(cate), len(cate)]), index=cate, columns=cate)

        for i in range(len(cate)):
            print("cate=", cate[i])
            for j in range(i+1, len(cate)):
                # i=1; j=2
                tmp = pd.crosstab(df[features[i]], df[features[j]])
                n = np.sum(tmp.values)
                m = min(tmp.shape)
                chi2 = chi2_contingency(tmp)[0]

                try:
                    if type == "contingency":
                        df_corr.iloc[i, j] = np.sqrt(chi2 / (n + chi2)) * np.sqrt(m / (m-1))
                    elif type == "cramersv":
                        df_corr.iloc[i, j] = np.sqrt(chi2 / (n * (m-1)))
                    else:
                        df_corr.iloc[i, j] = None
                except:
                    df_corr.iloc[i, j] = None
                df_corr.iloc[j, i] = df_corr.iloc[i, j]
        d_new_names = dict(zip(df_corr.columns.values,
                               df_corr.columns.values + " (" +
                               df[df_corr.columns.values].nunique().astype("str").values + ")"))
        df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    # All metric variables
    if len(metr):
        df_corr = abs(df[metr].corr(method="spearman"))
        d_new_names = dict(zip(df_corr.columns.values,
                               df_corr.columns.values + " (NA: " +
                               (df[df_corr.columns.values].isnull().mean() * 100).round(1).astype("str").values + "%)"))
        df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    # Filter out rows or cols below cutoff
    i_bool = (df_corr.max(axis=1) > cutoff).values
    df_corr = df_corr.loc[i_bool, i_bool]
    np.fill_diagonal(df_corr.values, 1)

    # Cluster df_corr
    new_order = df_corr.columns.values[fcluster(ward(1 - np.triu(df_corr)), n_cluster, criterion='maxclust').argsort()]
    df_corr = df_corr.loc[new_order, new_order]

    # Plot
    fig, ax = plt.subplots(1, 1)
    ax_act = ax
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Blues", ax=ax_act)
    ax_act.set_yticklabels(labels=ax_act.get_yticklabels(), rotation=0)
    ax_act.set_xticklabels(labels=ax_act.get_xticklabels(), rotation=90)
    if len(metr):
        ax_act.set_title("Absolute spearman correlation (cutoff at " + str(cutoff) +")")
    if len(cate):
        if type == "contingency":
            ax_act.set_title("Contingency coefficient (cutoff at " + str(cutoff) + ")")
        if type == "cramersv":
            ax_act.set_title("Cramer's V (cutoff at " + str(cutoff) + ")")
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()




# Plot CV results
def plot_cvresult(cv_results_, metric, x_var, color_var=None, column_var=None, row_var=None, style_var=None, pdf=None):
    df_cvres = pd.DataFrame.from_dict(cv_results_)
    df_cvres.columns = df_cvres.columns.str.replace("param_","")
    plt = sns.FacetGrid(df_cvres, col=column_var, row=row_var, margin_titles=True) \
        .map(sns.lineplot, x_var, "mean_test_" + metric,  # do not specify x= and y=!
             hue="#" + df_cvres[color_var].astype('str') if color_var is not None else None,
             style=df_cvres[style_var] if style_var is not None else None,
             marker="o").add_legend()
    if pdf is not None:
        plt.savefig(pdf)


# Plot generalization gap
def plot_gengap(cv_results_, metric, x_var, color_var=None, column_var=None, row_var=None, pdf=None):

    if pdf is not None:
        pdf_pages = PdfPages(pdf)

    # Prepare
    df_cvres = pd.DataFrame.from_dict(cv_results_)
    df_cvres.columns = df_cvres.columns.str.replace("param_","")
    df_gengap = df_cvres\
        .rename(columns={"mean_test_" + metric: "test",
                         "mean_train_" + metric: "train"}) \
        .assign(train_test_score_diff=lambda x: x.train - x.test) \
        .reset_index(drop=True)
    df_hlp = pd.melt(df_gengap,
                     id_vars=np.setdiff1d(df_gengap.columns.values, ["test", "train"]),
                     value_vars=["test", "train"],
                     var_name="fold", value_name="score")

    # Plot train vs test
    sns.FacetGrid(df_hlp, col=column_var, row=row_var,
                  margin_titles=True, height=5) \
        .map(sns.lineplot, x_var, "score",
             hue="#" + df_hlp[color_var].astype('str') if color_var is not None else None,
             style=df_hlp["fold"],
             marker="o").add_legend().set_ylabels(metric)
    if pdf is not None:
        pdf_pages.savefig()

    # Diff plot
    sns.FacetGrid(df_gengap, col=column_var, row=row_var,
                  margin_titles=True, height=5) \
        .map(sns.lineplot, x_var, "train_test_score_diff",
             hue="#" + df_gengap[color_var].astype('str') if color_var is not None else None,
             marker="o").add_legend().set_ylabels(metric + "_diff (train vs. test)")

    if pdf is not None:
        pdf_pages.savefig()
        pdf_pages.close()


# Plot model comparison
def plot_modelcomp(df_modelcomp_result, modelvar="model", runvar="run", scorevar="test_score", pdf=None):
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=df_modelcomp_result, x=modelvar, y=scorevar, showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                ax=ax)
    sns.lineplot(data=df_modelcomp_result, x=modelvar, y=scorevar,
                 hue="#" + df_modelcomp_result[runvar].astype("str"), linewidth=0.5, linestyle=":",
                 legend=None, ax=ax)
    if pdf is not None:
        fig.savefig(pdf)


# Plot the learning curve
def plot_learning_curve(n_train, score_train, score_test, pdf=None):
    df_lc_result = pd.DataFrame(zip(n_train, score_train[:, 0], score_test[:, 0]),
                                columns=["n_train", "train", "test"])\
        .melt(id_vars="n_train", value_vars=["train", "test"], var_name="fold", value_name="score")

    # Plot learning curve
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x="n_train", y="score", hue="fold", data=df_lc_result, marker="o", ax=ax)
    if pdf is not None:
        fig.savefig(pdf)


# Plot ML-algorithm performance
def plot_all_performances(y, yhat, labels=None, target_type="CLASS", regplot=True,
                          color=None, ylim=None, n_bins=10, w=18, h=12, pdf=None):
    # y=df_test["target"]; yhat=yhat_test; ylim = None; w=12; h=8

    if target_type == "CLASS":
        fig, ax = plt.subplots(2, 3)

        # Roc curve
        ax_act = ax[0, 0]
        fpr, tpr, cutoff = roc_curve(y, yhat[:, 1])
        roc_auc = roc_auc_score(y, yhat[:, 1])
        #sns.lineplot(fpr, tpr, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        ax_act.plot(fpr, tpr)
        props = {'xlabel': r"fpr: P($\^y$=1|$y$=0)",
                 'ylabel': r"tpr: P($\^y$=1|$y$=1)",
                 'title': "ROC (AUC = {0:.2f})".format(roc_auc)}
        ax_act.set(**props)

        # Confusion matrix
        ax_act = ax[0, 1]
        df_conf = pd.DataFrame(confusion_matrix(y, np.where(yhat[:, 1] > 0.5, 1, 0)))
        acc = accuracy_score(y, np.where(yhat[:, 1] > 0.5, 1, 0))
        sns.heatmap(df_conf, annot=True, fmt=".5g", cmap="Greys", ax=ax_act)
        props = {'xlabel': "Predicted label",
                 'ylabel': "True label",
                 'title': "Confusion Matrix (Acc ={0: .2f})".format(acc)}
        ax_act.set(**props)

        # Distribution plot
        ax_act = ax[0, 2]
        sns.distplot(yhat[:, 1][y == 1], color="red", label="1", bins=20, ax=ax_act)
        sns.distplot(yhat[:, 1][y == 0], color="blue", label="0", bins=20, ax=ax_act)
        props = {'xlabel': r"Predictions ($\^y$)",
                 'ylabel': "Density",
                 'title': "Distribution of Predictions",
                 'xlim': (0, 1)}
        ax_act.set(**props)
        ax_act.legend(title="Target", loc="best")

        # Calibration
        ax_act = ax[1, 0]
        true, predicted = calibration_curve(y, yhat[:, 1], n_bins=5)
        #sns.lineplot(predicted, true, ax=ax_act, marker="o")
        ax_act.plot(predicted, true, "o-")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Precision Recall
        ax_act = ax[1, 1]
        prec, rec, cutoff = precision_recall_curve(y, yhat[:, 1])
        prec_rec_auc = average_precision_score(y, yhat[:, 1])
        #sns.lineplot(rec, prec, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        ax_act.plot(rec, prec)
        props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Recall Curve (AUC = {0:.2f})".format(prec_rec_auc)}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (rec[i_thres], prec[i_thres]), fontsize=10)

        # Precision
        ax_act = ax[1, 2]
        pct_tested = np.array([])
        for thres in cutoff:
            pct_tested = np.append(pct_tested, [np.sum(yhat[:, 1] >= thres)/len(yhat)])
        sns.lineplot(pct_tested, prec[:-1], ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': "% Samples Tested",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Curve"}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (pct_tested[i_thres], prec[i_thres]), fontsize=10)

    if target_type == "REGR":
        fig, ax = plt.subplots(2, 3)

        def plot_scatter(x, y, xlabel="x", ylabel="y", title=None, ylim=None, ax_act=None):
            if ylim is not None:
                ax_act.set_ylim(ylim)
                tmp_scale = (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
            else:
                tmp_scale = 1
            tmp_cmap = mcolors.LinearSegmentedColormap.from_list("wh_bl_yl_rd",
                                                                [(1, 1, 1, 0), "blue", "yellow", "red"])
            p = ax_act.hexbin(x, y,
                              gridsize=(int(50 * tmp_scale), 50),
                              cmap=tmp_cmap)
            plt.colorbar(p, ax=ax_act)
            if regplot == True:
                sns.regplot(x, y, lowess=True, scatter=False, color="black", ax=ax_act)
            ax_act.set_title(title)
            ax_act.set_ylabel(ylabel)
            ax_act.set_xlabel(xlabel)

            ax_act.set_facecolor('white')
            # ax_act.grid(False)

            ylim = ax_act.get_ylim()
            xlim = ax_act.get_xlim()

            # Inner Histogram on y
            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
            inset_ax.set_axis_off()
            ax_act.get_shared_y_axes().join(ax_act, inset_ax)
            sns.distplot(y, color="grey", vertical=True, ax=inset_ax)

            # Inner-inner Boxplot on y
            xlim_inner = inset_ax.get_xlim()
            inset_ax.set_xlim(xlim_inner[0] - 0.3 * (xlim_inner[1] - xlim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 0.2, 1])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_y_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(y, palette=["grey"], orient="v", ax=inset_inset_ax)

            # Inner Histogram on x
            ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
            inset_ax.set_axis_off()
            ax_act.get_shared_x_axes().join(ax_act, inset_ax)
            sns.distplot(x, color="grey", ax=inset_ax)

            # Inner-inner Boxplot on x
            ylim_inner = inset_ax.get_ylim()
            inset_ax.set_ylim(ylim_inner[0] - 0.3 * (ylim_inner[1] - ylim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 1, 0.2])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_x_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(x, palette=["grey"], ax=inset_inset_ax)

            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))  # need to set again

        # Scatter plots
        plot_scatter(yhat, y,
                     xlabel=r"$\^y$", ylabel="y",
                     title=r"Observed vs. Fitted ($\rho_{Spearman}$ = " +
                           str(spear(y, yhat).round(3)) + ")",
                     ylim=ylim, ax_act=ax[0, 0])
        plot_scatter(yhat, y - yhat,
                     xlabel=r"$\^y$", ylabel=r"y-$\^y$", title="Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 0])
        plot_scatter(yhat, abs(y - yhat),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|", title="Absolute Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 1])
        plot_scatter(yhat, abs(y - yhat) / abs(y),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|/|y|", title="Relative Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 2])

        # Calibration
        ax_act = ax[0, 1]
        df_calib = pd.DataFrame({"y": y, "yhat": yhat})\
            .assign(bin=lambda x: pd.qcut(x["yhat"], 10, duplicates="drop").astype("str"))\
            .groupby(["bin"], as_index=False).agg("mean")\
            .sort_values("yhat")
        sns.lineplot("yhat", "y", data=df_calib, ax=ax_act, marker="o")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Distribution with ...
        ax_act = ax[0, 2]
        sns.distplot(y, color="blue", label="y", ax=ax_act)
        sns.distplot(yhat, color="red", label=r"$\^y$", ax=ax_act)
        ax_act.set_ylabel("density")
        ax_act.set_xlabel("")
        ax_act.set_title("Distribution")

        # ... inner boyxplot
        ylim = ax_act.get_ylim()
        ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
        inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
        inset_ax.set_axis_off()
        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
        df_distr = pd.concat([pd.DataFrame({"type": "y", "values": y}),
                              pd.DataFrame({"type": "yhat", "values": yhat})])
        sns.boxplot(x=df_distr["values"],
                    y=df_distr["type"].astype("category"),
                    # order=df[feature_act].value_counts().index.values[::-1],
                    palette=["blue", "red"],
                    ax=inset_ax)
        ax_act.legend(title="", loc="best")

    if target_type == "MULTICLASS":
       # y = df_test["target"]; yhat = yhat_test; labels = labels; target_type = TARGET_TYPE;
       # color = threecol; ylim = None; n_bins=10
        fig, ax = plt.subplots(2, 3)

        # AUC
        ax_act = ax[0, 0]
        k = yhat.shape[1]
        aucs = np.array([roc_auc_score(np.where(y == i, 1, 0), yhat[:, i]).round(2) for i in np.arange(k)])

        for i in np.arange(k):
            #  i=3
            y_bin = np.where(y == i, 1, 0)
            fpr, tpr, cutoff = roc_curve(y_bin, yhat[:, i])
            new_label = labels[i] + " (" + str(aucs[i]) + ")"
            ax_act.plot(fpr, tpr, color=color[i], label=new_label)
        mean_auc = np.average(aucs).round(3)
        weighted_auc = np.average(aucs, weights=np.array(np.unique(y, return_counts=True))[1, :]).round(3)
        props = dict(xlabel=r"fpr: P($\^y$=1|$y$=0)",
                     ylabel=r"tpr: P($\^y$=1|$y$=1)",
                     title="ROC\n" + r"($AUC_{mean}$ = " + str(mean_auc) + r", $AUC_{weighted}$ = " +
                           str(weighted_auc) + ")")
        ax_act.set(**props)
        ax_act.legend(title=r"Target ($AUC_{OvR}$)", loc='best')

        # Calibration
        #fig, ax = plt.subplots(2, 2)
        ax_act = ax[1, 0]
        for i in np.arange(k):
            true, predicted = calibration_curve(np.where(y == i, 1, 0), yhat[:, i], n_bins=n_bins, strategy="quantile")
            ax_act.plot(predicted, true, "o-", color=color[i], label=labels[i], markersize=4)
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)
        ax_act.legend(title="Target", loc='best')

        # Confusion Matrix
        ax_act = ax[0, 1]

        y_pred = yhat.argmax(axis=1)
        freq_true = np.unique(y, return_counts=True)[1]
        freqpct_true = np.round(np.divide(freq_true, len(y)) * 100, 1)
        freq_pred = np.unique(y_pred, return_counts=True)[1]
        freqpct_pred = np.round(np.divide(freq_pred, len(y)) * 100, 1)

        m_conf = confusion_matrix(y, y_pred)
        ylabels = [labels[i] + " (" + str(freq_true[i]) + ": " + str(freqpct_true[i]) + "%)" for i in
                   np.arange(len(labels))]
        xlabels = [labels[i] + " (" + str(freq_pred[i]) + ": " + str(freqpct_pred[i]) + "%)" for i in
                   np.arange(len(labels))]
        df_conf = pd.DataFrame(m_conf, columns=labels, index=labels).rename_axis(index="True label",
                                                                                 columns="Predicted label")
        acc = accuracy_score(y, y_pred)
        sns.heatmap(df_conf, annot=True, fmt=".5g", cmap="Blues", ax=ax_act,
                    xticklabels=True, yticklabels=True, cbar=False)
        ax_act.set_yticklabels(labels=ylabels, rotation=0)
        ax_act.set_xticklabels(labels=xlabels, rotation=90, ha="center")
        props = dict(ylabel="True label (#: %)",
                     xlabel="Predicted label (#: %)",
                     title="Confusion Matrix (Acc ={0: .2f})".format(acc))
        ax_act.set(**props)
        for text in ax_act.texts[::len(labels) + 1]:
            text.set_weight('bold')

        # Barplots
        ax_act = ax[0, 2]
        df_conf.iloc[::-1].plot.barh(stacked=True, ax=ax_act, color=color[:len(labels)])
        ax_act.legend(title="Predicted label", loc='center left', bbox_to_anchor=(1, 0.5))
        #fig, ax = plt.subplots(2, 2)
        ax_act = ax[1, 1]
        df_conf.copy().T.iloc[:,::-1].plot.bar(stacked=True, ax=ax_act, color=color[:len(labels)][::-1])
        handles, labels = ax_act.get_legend_handles_labels()
        ax_act.legend(handles[::-1], labels[::-1], title="True label", loc='center left', bbox_to_anchor=(1, 0.5))

        # Metrics
        #fig, ax = plt.subplots(2, 3)
        ax_act = ax[1, 2]
        prec = np.round(np.diag(m_conf) / m_conf.sum(axis=0) * 100, 1)
        rec = np.round(np.diag(m_conf) / m_conf.sum(axis=1) * 100, 1)
        f1 = np.round(2 * prec * rec / (prec + rec), 1)
        df_metrics = pd.DataFrame(np.column_stack((y, np.flip(np.argsort(yhat, axis=1), axis=1)[:, :3])),
                                  columns=["y", "yhat1", "yhat2", "yhat3"]). \
            assign(acc_top1=lambda x: (x["y"] == x["yhat1"]).astype("int"),
                   acc_top2=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"])).astype("int"),
                   acc_top3=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"]) |
                                       (x["y"] == x["yhat3"])).astype("int")). \
            assign(label=lambda x: np.array(labels, dtype="object")[x["y"].values]). \
            groupby(["label"])["acc_top1", "acc_top2", "acc_top3"].agg("mean").round(2). \
            join(pd.DataFrame(np.stack((aucs, rec, prec, f1), axis=1),
                              index=labels, columns=["auc", "recall", "precision", "f1"]))
        sns.heatmap(df_metrics.T, annot=True, fmt=".5g",
                    cmap=ListedColormap(['white']), linewidths=2, linecolor="black", cbar=False,
                    ax=ax_act, xticklabels=True, yticklabels=True)
        ax_act.set_yticklabels(labels=['Accuracy\n Top1', 'Accuracy\n Top2', 'Accuracy\n Top3', "AUC\n 1-vs-all",
                                       'Recall\n' r"P($\^y$=k|$y$=k))", 'Precision\n' r"P($y$=k|$\^y$=k))", 'F1'])
        #ax_act.set_yticklabels(labels=ylabels, rotation=0)
        ax_act.xaxis.tick_top()  # x axis on top
        ax_act.xaxis.set_label_position('top')
        ax_act.tick_params(left=False, top=False)
        ax_act.set_xlabel("True label")
        #ax_act.patch.set_edgecolor('black')
        #ax_act.patch.set_linewidth('1')

    # Adapt figure
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()


# Variable importance
def calc_varimp_by_permutation(df, df_ref, fit,
                               target, metr, cate,
                               target_type="CLASS",
                               b_sample=None, b_all=None,
                               features=None,
                               random_seed=999,
                               n_jobs=4):
    # df=df_train;  df_ref=df; target = "target"
    #pdb.set_trace()
    all_features = np.append(metr, cate)
    if features is None:
        features = all_features

    # Original performance
    if target_type in ["CLASS", "MULTICLASS"]:
        perf_orig = auc(df[target],
                        scale_predictions(fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df)),
                                          b_sample, b_all))
    elif target_type == "REGR":
        perf_orig = spear(df[target],
                          fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df)))

    # Performance per variable after permutation
    np.random.seed(random_seed)
    i_perm = np.random.permutation(np.arange(len(df)))  # permutation vector

    # TODO: add all variables used as parameter
    def run_in_parallel(df, feature):
        df_perm = df.copy()
        df_perm[feature] = df_perm[feature].values[i_perm]
        if target_type in ["CLASS", "MULTICLASS"]:
            perf = auc(df_perm[target],
                       scale_predictions(fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref)
                                                           .fit_transform(df_perm)),
                                         b_sample, b_all))
        elif target_type == "REGR":
            perf = spear(df_perm[target],
                         fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df_perm)))
        return perf
    perf = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(df, feature)
                                                      for feature in features)

    # Collect performances and calculate importance
    df_varimp = pd.DataFrame({"feature": features, "perf_diff": np.maximum(0, perf_orig - perf)})\
        .sort_values(["perf_diff"], ascending=False).reset_index(drop=False)\
        .assign(importance=lambda x: 100 * x["perf_diff"] / max(x["perf_diff"]))\
        .assign(importance_cum=lambda x: 100 * x["perf_diff"].cumsum() / sum(x["perf_diff"]))\
        .assign(importance_sumnormed=lambda x: 100 * x["perf_diff"] / sum(x["perf_diff"]))

    return df_varimp


# Plot variable importance
def plot_variable_importance(df_varimp, mask=None, w=18, h=12, pdf=None):
    # Prepare
    n_features = len(df_varimp)
    if mask is not None:
        df_varimp = df_varimp[mask]

    # Plot
    fig, ax = plt.subplots(1, 1)
    sns.barplot("importance", "feature", hue="Category", data=df_varimp,
                dodge=False, palette=sns.xkcd_palette(["blue", "orange", "red"]), ax=ax)
    ax.legend(loc=8, title="Importance")
    ax.plot("importance_cum", "feature", data=df_varimp, color="grey", marker="o")
    ax.set_xlabel(r"importance / cumulative importance in % (-$\bullet$-)")
    ax.set_title("Top{0: .0f} (of{1: .0f}) Feature Importances".format(len(df_varimp), n_features))
    fig.tight_layout()
    if pdf:
        fig.savefig(pdf)


# Partial dependence
def calc_partial_dependence(df, df_ref, fit,
                            target, metr, cate,
                            target_type="CLASS",
                            b_sample=None, b_all=None,
                            features=None,
                            quantiles=np.arange(0, 1.1, 0.1),
                            n_jobs=4):
    # df=df_test;  df_ref=df_traintest; target = "target"; target_type=TARGET_TYPE; features=None;
    # quantiles = np.arange(0, 1.1, 0.1);n_jobs=4
    all_features = np.append(metr, cate)
    if features is None:
        features = all_features

    df_pd = pd.DataFrame()

    def run_in_parallel(feature):
        # feature = features[3]
        if feature in metr:
            values = df_ref[feature].quantile(quantiles)
        else:
            values = np.sort(df_ref[feature].unique())

        df_tmp = df.copy()  # save original data

        df_pd_feature = pd.DataFrame()
        for value in values:
            #value=values[0]
            df_tmp[feature] = value
            if target_type == "CLASS":
                yhat_mean = np.mean(scale_predictions(fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref).
                                                                        fit_transform(df_tmp)),
                                    b_sample, b_all)[:, 1])
            if target_type == "REGR":
                yhat_mean = np.mean(fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df_tmp)),
                                    b_sample, b_all)

            df_pd_feature = pd.concat([df_pd_feature,
                                       pd.DataFrame({"feature": feature, "value": str(value), "yhat_mean": yhat_mean},
                                                    index=[0])])
        return df_pd_feature
    df_pd = pd.concat(Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(feature)
                                                                 for feature in features))
    return df_pd


# --- Classes ----------------------------------------------------------------------------------------

'''
# Special Kfold splitter: training fold only from training data, test fold only from test data
class KFoldTraintestSep:
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, folds=None):
        for i_train, i_test in KFold(n_splits=self.n_splits,
                                     shuffle=True,
                                     random_state=self.random_state).split(X, y):
            i_train_train = i_train[folds[i_train] == "train"]
            i_test_test = i_test[folds[i_test] == "test"]
            yield i_train_train, i_test_test

    def get_n_splits(self):
        return self.n_splits
'''

# Special splitter: training fold only from training data, test fold only from test data
class TrainTestSep:
    def __init__(self, n_splits=1, sample_type="cv", fold_var="fold", random_state=42):
        self.n_splits = n_splits
        self.sample_type = sample_type
        self.fold_var = fold_var
        self.random_state = random_state

    def split(self, df):
        i_df = np.arange(len(df))
        np.random.seed(self.random_state)
        np.random.shuffle(i_df)
        i_train = i_df[df[self.fold_var].values[i_df] == "train"]
        i_test = i_df[df[self.fold_var].values[i_df] == "test"]
        if self.sample_type == "cv":
            splits_train = np.array_split(i_train, self.n_splits)
            splits_test = np.array_split(i_test, self.n_splits)
        else:
            splits_train = None
            splits_test = None
        for i in range(self.n_splits):
            if self.sample_type == "cv":
                i_train_yield = np.concatenate(splits_train)
                if self.n_splits > 1:
                    i_train_yield = np.setdiff1d(i_train_yield, splits_train[i], assume_unique=True)
                i_test_yield = splits_test[i]
            elif self.sample_type == "bootstrap":
                np.random.seed(self.random_state * (i + 1))
                i_train_yield = np.random.choice(i_train, len(i_train))
                np.random.seed(self.random_state * (i + 1))
                i_test_yield = np.random.choice(i_test, len(i_test))
            else:
                i_train_yield = None
                i_test_yield = None
            yield i_train_yield, i_test_yield

    def get_n_splits(self):
        return self.n_splits


# Map Nonexisting members of a string column to modus
class MapNonexisting(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self._d_unique = None
        self._d_modus = None

    def fit(self, df):
        self._d_unique = {x: pd.unique(df[x]) for x in self.features}
        self._d_modus = {x: df[x].value_counts().index[0] for x in self.features}
        return self

    def transform(self, df):
        df = df.apply(lambda x: x.where(np.in1d(x, self._d_unique[x.name]),
                                        self._d_modus[x.name]) if x.name in self.features else x)
        return df

    def fit_transform(self, df, y=None, **fit_params):
        if fit_params["transform"]:
            return self.fit(df).transform(df)
        else:
            self.fit(df)
            return df


# Map Non-topn frequent members of a string column to "other" label
class MapToomany(BaseEstimator, TransformerMixin):
    def __init__(self, features, n_top=10, other_label="_OTHER_"):
        self.features = features
        self.other_label = other_label
        self.n_top = n_top
        self._s_levinfo = None
        self._toomany = None
        self._d_top = None
        self._statistics = None

    def fit(self, df, y=None):
        self._s_levinfo = df[self.features].apply(lambda x: x.unique().size).sort_values(ascending=False)
        self._toomany = self._s_levinfo[self._s_levinfo > self.n_top].index.values
        self._d_top = {x: df[x].value_counts().index.values[:self.n_top] for x in self._toomany}
        self._statistics = {"_s_levinfo": self._s_levinfo, "_toomany": self._toomany, "_d_top": self._d_top}
        return self

    def transform(self, df):
        df = df.apply(lambda x: x.where(np.in1d(x, self._d_top[x.name]),
                                        other=self.other_label) if x.name in self._toomany else x)
        return df


# Target Encoding
class TargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, features, encode_flag_column="use_for_encoding", target="target"):
        self.features = features
        self.encode_flag_column = encode_flag_column
        self.target = target
        self._d_map = None
        self._statistics = None

    def fit(self, df, y=None):
        if df[self.target].nunique() > 2:
            # Take majority class in case of MULTICLASS target
            df["tmp"] = np.where(df[self.target] == df[self.target].value_counts().values[0], 1, 0)
        else:
            df["tmp"]=df[self.target]
        self._d_map = {x: df.loc[df[self.encode_flag_column] == 1, :].reset_index(drop=True)
                            .groupby(x, as_index=False)["tmp"].agg("mean")
                            .sort_values("tmp", ascending=False)
                            .assign(rank=lambda x: np.arange(len(x)) + 1)
                            .set_index(x)
                            ["rank"]
                            .to_dict() for x in self.features}
        self._statistics = {"_d_map": self._d_map}
        return self

    def transform(self, df):
        #pdb.set_trace()
        df[self.features + "_ENCODED"] = df[self.features].apply(lambda x: x.map(self._d_map[x.name])
                                                                 .fillna(np.median(list(self._d_map[x.name].values()))))
        if self.encode_flag_column in df.columns.values:
            #return df.loc[df[self.encode_flag_column] != 1, :].reset_index(drop=True) #TODO: make this parameter-depen.
            return df
        else:
            return df


# SimpleImputer for data frames
class DfSimpleImputer(SimpleImputer):
    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self.features = features

    def fit(self, df, y=None, **kwargs):
        if len(self.features):
            fit = super().fit(df[self.features], **kwargs)
            return fit
        else:
            return self

    def transform(self, df):
        if len(self.features):
            df[self.features] = super().transform(df[self.features].values)
        return df


# Random Imputer
class DfRandomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, df_ref=None):
        self.features = features
        self.df_ref = df_ref

    def fit(self, df, y=None):
        if self.df_ref is None:
            self.df_ref = df
        return self

    def transform(self, df=None, y=None):
        #pdb.set_trace()
        if len(self.features):
            # for feature in self.features:
            #     if df[feature].isnull().any():
            #         df[feature] = np.where(np.isnan(df[feature]),
            #                                self.df_ref[feature]
            #                                .dropna()
            #                                .sample(n=len(df[feature]), replace=True,
            #                                        random_state=np.where(feature == self.features)[0][0]),
            #                                df[feature])
            df[self.features] = df[self.features].apply(lambda x:
                                                        np.where(np.isnan(x),
                                                                 self.df_ref[x.name]
                                                                 .dropna()
                                                                 .sample(n=len(x), replace=True,
                                                                         random_state=
                                                                         np.where(x.name == self.features)[0][0]),
                                                                 x))
        return df


# Convert
class Convert(BaseEstimator, TransformerMixin):
    def __init__(self, features, convert_to):
        self.features = features
        self.convert_to = convert_to

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if len(self.features):
            df[self.features] = df[self.features].astype(self.convert_to)
            if self.convert_to == "str":
                df[self.features] = df[self.features].replace("nan", np.nan)
        return df


# Winsorize
class Winsorize(BaseEstimator, TransformerMixin):
    def __init__(self, features, fresh=False, lower_quantile=None, upper_quantile=None):
        self.features = features
        self.fresh = fresh
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self._d_lower = None
        self._d_upper = None

    def fit(self, df, y=None):
        if not self.fresh:
            if self.lower_quantile is not None:
                self._d_lower = df[self.features].quantile(self.lower_quantile).to_dict()
            if self.upper_quantile is not None:
                self._d_upper = df[self.features].quantile(self.upper_quantile).to_dict()
        return self

    def transform(self, df):
        if self.fresh:
            if self.lower_quantile is not None:
                self._d_lower = df[self.features].quantile(self.lower_quantile).to_dict()
            if self.upper_quantile is not None:
                self._d_upper = df[self.features].quantile(self.upper_quantile).to_dict()
        if self.lower_quantile is not None:
            df[self.features] = df[self.features].apply(lambda x: x.clip(lower=self._d_lower[x.name]))
        if self.upper_quantile is not None:
            df[self.features] = df[self.features].apply(lambda x: x.clip(upper=self._d_upper[x.name]))
        return df


# Zscale
class Zscale(BaseEstimator, TransformerMixin):
    def __init__(self, features, fresh=False):
        self.features = features
        self.fresh = fresh
        self._d_mean = None
        self._d_std = None

    def fit(self, df, y=None):
        if not self.fresh:
            self._d_mean = df[self.features].mean().to_dict()
            self._d_std = df[self.features].std().to_dict()
        return self

    def transform(self, df):
        if self.fresh:
            self._d_mean = df[self.features].mean().to_dict()
            self._d_std = df[self.features].std().to_dict()
        df[self.features] = df[self.features].apply(lambda x: (x - self._d_mean[x.name]) / self._d_std[x.name])
        return df


# Binning: TODO: Save borders and apply in transform
class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.fresh = True

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        #pdb.set_trace()
        df[self.features] = df[self.features].apply(lambda x: char_bins(x))
        return df


# Undersample
class Undersample(BaseEstimator, TransformerMixin):
    def __init__(self, n_max_per_level, random_state=42):
        self.n_max_per_level = n_max_per_level
        self.random_state = random_state
        self.b_sample = None
        self.b_all = None

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df

    def fit_transform(self, df, y=None, target="target"):
        # pdb.set_trace()
        self.b_all = df[target].value_counts().values / len(df)
        df = df.groupby(target).apply(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]),
                                                         random_state=self.random_state)) \
            .reset_index(drop=True)\
            .sample(frac=1).reset_index(drop=True)
        self.b_sample = df[target].value_counts().values / len(df)
        return df


class FeatureEngineeringTitanic(BaseEstimator, TransformerMixin):
    def __init__(self, derive_deck=True, derive_familysize=True, derive_fare_pp=True):
        self.derive_deck = derive_deck
        self.derive_familysize = derive_familysize
        self.derive_fare_pp = derive_fare_pp

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if self.derive_deck:
            df["deck"] = df["cabin"].str[:1]
        if self.derive_familysize:
            df["familysize"] = df["sibsp"].astype("int") + df["parch"].astype("int") + 1
        if self.derive_fare_pp:
            df["fare_pp"] = df.groupby("ticket")["fare"].transform("mean")
        return df


# Create sparse matrix
class CreateSparseMatrix(BaseEstimator, TransformerMixin):
    def __init__(self, metr=None, cate=None, df_ref=None, sparse=True):
        self.metr = metr
        self.cate = cate
        self.df_ref = df_ref
        self.sparse = sparse
        self._d_categories = None

    def fit(self, df=None, y=None):
        if self.df_ref is None:
            self.df_ref = df
        if self.cate is not None and len(self.cate) > 0:
            self._d_categories = [self.df_ref[x].unique() for x in self.cate]
        return self

    def transform(self, df=None, y=None):
        if self.metr is not None and len(self.metr) > 0:
            if self.sparse:
                # m_metr = df[self.metr].to_sparse().to_coo()
                m_metr = df[self.metr].astype(pd.SparseDtype("float", np.nan))
            else:
                m_metr = df[self.metr].to_numpy()
        else:
            m_metr = None
        if self.cate is not None and len(self.cate) > 0:
            enc = OneHotEncoder(categories=self._d_categories, sparse=self.sparse)

            m_cate = enc.fit_transform(df[self.cate], y)
            # if len(self.cate) == 1:
            #     m_cate = enc.fit_transform(df[self.cate].reshape(-1, 1), y)
            # else:
            #     m_cate = enc.fit_transform(df[self.cate], y)
        else:
            m_cate = None
        if self.sparse:
            return hstack([m_metr, m_cate], format="csr")
        else:
            return np.hstack([m_metr, m_cate])


# # NearestCentroid fails in Classifier context
# class NearestCentroidClassifier(NearestCentroid):
#     def __init__(self, metric='euclidean', shrink_threshold=None):
#         super().__init__(metric=metric, shrink_threshold=shrink_threshold)
#
#     def predict_proba(self, X):
#         # pdb.set_trace()
#         tmp = self.predict(X)
#         return np.column_stack((tmp, tmp))

# # GLMNET Classifier: ONLY on Unix
# class glmnetClassifier(BaseEstimator, ClassifierMixin):
#
#     def __init__(self, lambdau=1, alpha=1):
#         self.lambdau = lambdau
#         self.alpha = alpha
#
#     def fit(self, X, y=None):
#         return glmnet(x=X, y=sp.array(y, dtype="float"), family="binomial", lambdau=self.lambdau, alpha=self.alpha)
#
#     def predict_proba(self, X, y=None):
#         glmnetPredict(self, newx=X, ptype='response')