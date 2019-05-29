
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from scipy.stats import chi2_contingency
from scipy.sparse import hstack

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *

# ######################################################################################################################
# My Functions
# ######################################################################################################################

# def setdiff(a, b):
#     return [x for x in a if x not in set(b)]


# def union(a, b):
#     return a + [x for x in b if x not in set(a)]

# Overview of values 
def create_values_df(df, topn):
    return pd.concat([df[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_c"})
                      for catname in df.select_dtypes(["object"]).columns.values], axis=1)


# Show closed figure again
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    

# Plot distribution regarding target
def plot_distr(df, features, target="target", varimp=None,
               ncol=1, nrow=1, w=8, h=6, pdf=None):
    # df = df; features = metr_toprint; target = "fold_num";  varimp=varimp_metr_fold;
    # ncol=2; nrow=2; pdf=None; w=8; h=6

    # Help variables
    n_ppp = ncol * nrow  # plots per page
    n_pages = len(features) // n_ppp + 1  # number of pages
    pdf_pages = None

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
                if df[feature_act].dtype == "object":
                    # Prepare data
                    df_plot = pd.DataFrame({"h": df.groupby(feature_act)[target].mean(),
                                            "w": df.groupby(feature_act).size()}).reset_index()
                    df_plot.w = df_plot.w/max(df_plot.w)
                    df_plot["new_w"] = np.where(df_plot["w"].values < 0.2, 0.2, df_plot["w"])

                    # Target barplot
                    # sns.barplot(df_tmp.h, df_tmp[cate[page * ppp + i]], orient="h", color="coral", ax=axact)
                    ax_act.barh(df_plot[feature_act], df_plot.h, height=df_plot.new_w, edgecolor="black")
                    ax_act.set_xlabel("Proportion Target = Y")
                    if varimp is not None:
                        ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                    ax_act.axvline(np.mean(df[target]), ls="dotted", color="black")

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
                    sns.distplot(df.loc[df[target] == 1, feature_act].dropna(), color="red", label="1", ax=ax_act)
                    sns.distplot(df.loc[df[target] == 0, feature_act].dropna(), color="blue", label="0", ax=ax_act)
                    # sns.FacetGrid(df, hue=target, palette=["red","blue"])\
                    #     .map(sns.distplot, metr[i])\
                    #     .add_legend() # does not work for multiple axes
                    if varimp is not None:
                        ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                    ax_act.set_ylabel("density")
                    ax_act.set_xlabel(feature_act + "(NA: " +
                                      str(df[feature_act].isnull().mean().round(3) * 100) +
                                      "%)")

                    # Boxplot
                    ylim = ax_act.get_ylim()
                    ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
                    inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
                    inset_ax.set_axis_off()
                    ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                    i_bool = df[feature_act].notnull()
                    sns.boxplot(x=df.loc[i_bool, feature_act],
                                y=df.loc[i_bool, target].astype("category"),
                                palette=["blue", "red"],
                                ax=inset_ax)
                    ax_act.legend(title=target, loc="best")
                # plt.show()
            else:
                ax_act.axis("off")  # Remove left-over plots
        # plt.subplots_adjust(wspace=1)
        fig.set_size_inches(w=w, h=h)
        fig.tight_layout()
        # l_fig.append(fig)
        if pdf is not None:
            pdf_pages.savefig(fig)
            # plt.close(fig)
        plt.show()
    if pdf is not None:
        pdf_pages.close()


# Plot correlation
def plot_corr(df, features, cutoff=0, w=8, h=6, pdf=None):
    # df = df; features = cate; cutoff = 0.1; w=8; h=6; pdf="blub.pdf"

    metr = features[df[features].dtypes != "object"]
    cate = features[df[features].dtypes == "object"]
    df_corr = None

    if len(metr) and len(cate):
        raise Exception('Mixed dtypes')
        # return

    if len(cate):
        df_corr = pd.DataFrame(np.ones([len(cate), len(cate)]), index=cate, columns=cate)
        for i in range(len(cate)):
            print("cate=", cate[i])
            for j in range(i+1, len(cate)):
                # i=1; j=2
                tmp = pd.crosstab(df[features[i]], df[features[j]])
                n = np.sum(tmp.values)
                m = min(tmp.shape)
                chi2 = chi2_contingency(tmp)[0]
                df_corr.iloc[i, j] = np.sqrt(chi2 / (n + chi2)) * np.sqrt(m / (m-1))
                df_corr.iloc[j, i] = df_corr.iloc[i, j]

    if len(metr):
        df_corr = abs(df[metr].corr(method="spearman"))

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
        # plt.close(fig)
    plt.show()


# Univariate variable importance
def calc_imp(df, features, target="target"):
    # df=df; features=metr; target="fold"
    varimp = pd.Series()
    for feature_act in features:
        # feature_act=metr[0]

        if df[feature_act].dtype == "object":
            varimp_act = {feature_act: (roc_auc_score(y_true=df[target].values,
                                                      y_score=df[[feature_act, target]]
                                                      .groupby(feature_act)[target]
                                                      .transform("mean").values)
                                        .round(3))}
        else:
            varimp_act = {feature_act: (roc_auc_score(y_true=df[target].values,
                                                      y_score=df[[target]]
                                                      .assign(dummy=pd.qcut(df[feature_act], 10).astype("object")
                                                              .fillna("(Missing)"))
                                                      .groupby("dummy")[target]
                                                      .transform("mean").values)
                                        .round(3))}
        varimp = varimp.append(pd.Series(varimp_act))
    varimp.sort_values(ascending=False, inplace=True)
    return varimp


# Undersample
def undersample_n(df, n_max_per_level, target="target", random_state=42):
    b_all = df[target].mean()
    df = df.groupby(target).apply(lambda x: x.sample(min(n_max_per_level, x.shape[0]),
                                                     random_state=random_state)) \
        .reset_index(drop=True)
    b_sample = df[target].mean()
    return df, b_sample, b_all


# Rescale predictions (e.g. to rewind undersampling)
def scale_predictions(yhat, b_sample=None, b_all=None):
    if b_sample is None:
        yhat_rescaled = yhat
    else:
        tmp = yhat * np.array([1 - b_all, b_all]) / np.array([1 - b_sample, b_sample])
        yhat_rescaled = (tmp.T / tmp.sum(axis=1)).T
    return yhat_rescaled


# Create sparse matrix
def create_sparse_matrix(df, metr=None, cate=None, df_ref=None):
    if metr is not None:
        m_metr = df[metr].to_sparse().to_coo()
    else:
        m_metr = None
    if cate is not None:
        if df_ref is None:
            enc = OneHotEncoder()
        else:
            enc = OneHotEncoder(categories=[df_ref[x].unique() for x in cate])
        if len(cate) == 1:
            m_cate = enc.fit_transform(df[cate].reshape(-1, 1))
        else:
            m_cate = enc.fit_transform(df[cate])
    else:
        m_cate = None
    return hstack([m_metr, m_cate], format="csr")


# Plot ML-algorithm performance
def plot_all_performances(y, yhat, w=12, h=8, pdf=None):
    # y=df_test["target"]; yhat=yhat_test; w=12; h=8
    fig, ax = plt.subplots(2, 3)

    # Roc curve
    ax_act = ax[0, 0]
    fpr, tpr, cutoff = roc_curve(y, yhat[:, 1])
    roc_auc = roc_auc_score(y, yhat[:, 1])
    sns.lineplot(fpr, tpr, ax=ax_act, color=sns.xkcd_rgb["red"])
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
    true, predicted = calibration_curve(y, yhat[:, 1], n_bins=10)
    sns.lineplot(predicted, true, ax=ax_act, color=sns.xkcd_rgb["red"], marker="o")
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    ax_act.set(**props)

    # Precision Recall
    ax_act = ax[1, 1]
    prec, rec, cutoff = precision_recall_curve(y, yhat[:, 1])
    prec_rec_auc = average_precision_score(y, yhat[:, 1])
    sns.lineplot(rec, prec, ax=ax_act, color=sns.xkcd_rgb["red"])
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
    sns.lineplot(pct_tested, prec[:-1], ax=ax_act, color=sns.xkcd_rgb["red"])
    props = {'xlabel': "% Samples Tested",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Curve"}
    ax_act.set(**props)
    for thres in np.arange(0.1, 1, 0.1):
        i_thres = np.argmax(cutoff > thres)
        ax_act.annotate("{0: .1f}".format(thres), (pct_tested[i_thres], prec[i_thres]), fontsize=10)

    # Adapt figure
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()


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
                    i_train_yield = np.setdiff1d(i_train_yield, splits_train[i])
                i_test_yield = splits_test[i]
            elif self.sample_type == "bootstrap":
                np.random.seed(self.random_state * (i+1))
                i_train_yield = np.random.choice(i_train, len(i_train))
                np.random.seed(self.random_state * (i+1))
                i_test_yield = np.random.choice(i_test, len(i_test))
            else:
                i_train_yield = None
                i_test_yield = None
            yield i_train_yield, i_test_yield

    def get_n_splits(self):
        return self.n_splits
