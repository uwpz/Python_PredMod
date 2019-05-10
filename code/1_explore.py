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

import os
os.getcwd()
# os.chdir("C:/My/Projekte/Python_PredMod")
# exec(open("./code/0_init.py").read())

exec(open("./code/0_init.py").read())


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
df["target"] = np.where(df.survived == 0, "N", "Y")
df["target_num"] = df.target.map({"N": 0, "Y": 1})
df[["target", "target_num"]].describe(include="all")

# Train/Test fold: usually split by time
df["fold"] = "train"
df.loc[df.sample(frac=0.3, random_state=123).index, "fold"] = "test"
print(df.fold.value_counts())

# Define the id
df["id"] = np.arange(df.shape[0]) + 1


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------
metr = df_meta_sub.loc[df_meta_sub.type == "metr", "variable"].values
df[metr] = df[metr].apply(pd.to_numeric)
df[metr].describe()


# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
metr_binned = metr + "_BINNED_"
df[metr_binned] = df[metr].apply(lambda x: pd.qcut(x, 10).astype(str))

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
fig, ax = plt.subplots(1, len(metr))
for i in range(len(metr)):
    # i=1
    axact = ax.flat[i]
    sns.distplot(df.loc[df.target == "Y", metr[i]].dropna(), color="red", label="Y", ax=axact)
    sns.distplot(df.loc[df.target == "N", metr[i]].dropna(), color="blue", label="N", ax=axact)
    axact.set_title(metr[i])
    axact.set_ylabel("density")
    axact.set_xlabel(metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
    ylim = axact.get_ylim()
    axact.set_ylim(ylim[0] - 0.3*(ylim[1]-ylim[0]))
    inset_ax = axact.inset_axes([0, 0, 1, 0.2])
    inset_ax.set_axis_off()
    axact.get_shared_x_axes().join(axact, inset_ax)
    sns.boxplot(x=df[metr[i]].dropna(), y=df["target"], palette={"Y": "red", "N": "blue"}, ax=inset_ax)
ax[0].legend(title="Target", loc="best")
fig.tight_layout()
# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
fig.set_size_inches(w=12, h=8)
fig.savefig(plotloc + "metr.pdf", format='pdf')
# plt.show(fig)
plt.close(fig)


# plotnine cannot plot several plots on one page
"""
i=1
nbins = 20
target_name = "target"
color = ["blue","red"]
levs_target = ["N","Y"]
p=(ggplot(data = df, mapping = aes(x = metr[i])) +
      geom_histogram(mapping = aes(y = "..density..", fill = target_name, color = target_name), 
                     stat = stat_bin(bins = nbins), position = "identity", alpha = 0.2) +
      geom_density(mapping = aes(color = target_name)) +
      scale_fill_manual(limits = levs_target[::-1], values = color[::-1], name = target_name) + 
      scale_color_manual(limits = levs_target[::-1], values = color[::-1], name = target_name) +
      labs(title = metr[i],
           x = metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
      )
p
plt.show()
plt.close()
"""

# Winsorize
df[metr] = df[metr].apply(lambda x: winsorize(x, (0.01, 0.01)))  # hint: plot again before deciding for log-trafo

# Log-Transform
tolog = np.array(["fare"], dtype="object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - max(0, np.min(x)) + 1))
np.place(metr, np.isin(metr, tolog), tolog + "_LOG_")  # adapt metadata (keep order)


# --- Final variable information ------------------------------------------------------------------------------------
# Univariate variable importance
varimp_metr = pd.Series({x: (roc_auc_score(y_true=df["target_num"].values,
                                           y_score=df[["target_num"]].
                                           assign(dummy=pd.qcut(df[x], 10).astype("object").fillna("(Missing)")).
                                           groupby("dummy")["target_num"].transform("mean").values)
                             .round(2))
                         for x in metr}).sort_values(ascending=False)
print(varimp_metr)

# Plot 
# plots = get_plot_distr_metr_class(df, metr, missinfo = misspct, varimpinfo = varimp)
# ggsave(paste0(plotloc, "titanic_distr_metr_final.pdf"), marrangeGrob(plots, ncol=4, nrow=2), width=18, height=12)


# --- Removing variables -------------------------------------------------------------------------------------------
# Remove leakage features
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED")  # keep "binned" version in sync

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
m_corr = abs(df[metr].corr(method="spearman"))
fig = sns.heatmap(m_corr, annot=True, fmt=".2f", cmap="Blues").get_figure()
fig.set_size_inches(w=6, h=6)
plt.close(fig)
fig.savefig(plotloc + "corr_metr.pdf")
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED")  # keep "binned" version in sync


"""
# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
(varimp_metr_fold = filterVarImp(df[metr], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))

# Plot: only variables with with highest importance
metr_toprint = names(varimp_metr_fold)[varimp_metr_fold >= cutoff_varimp]
plots = map(metr_toprint, ~ BoxCore::plot_distr(df[[.]], df$fold_test, ., "fold_test", varimps = varimp_metr_fold,
                                                colors = c("blue","red")))
ggsave(paste0(plotloc, TYPE, "_distr_metr_final_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2),
       width = 18, height = 12)
"""


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
df[cate] = df[cate].astype("object")
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
varimp_cate = pd.Series({x: (roc_auc_score(y_true=df.target_num.values,
                                           y_score=df[[x, "target_num"]].groupby(x)["target_num"].
                                           transform("mean").values)
                             .round(2))
                         for x in cate}).sort_values(ascending=False)
print(varimp_cate)


# Check
ncol = 2
nrow = 2
ppp = ncol * nrow

pdf_pages = PdfPages(plotloc + "cate.pdf")
for page in range(len(cate) % ppp + 1):
    fig, ax = plt.subplots(ncol, nrow)
    for i in range(ppp):
        if page * ppp + i <= max(range(len(cate))):
            # fig, ax = plt.subplots(1, 2); page=1; i=1
            df_tmp = pd.DataFrame({"h": df.groupby(cate[page * ppp + i])["target_num"].mean(),
                                   "w": df.groupby(cate[page * ppp + i]).size()}).reset_index()
            df_tmp.w = df_tmp.w/max(df_tmp.w)
            df_tmp["new_w"] = np.where(df_tmp["w"].values < 0.2, 0.2, df_tmp["w"])
            axact = ax.flat[i]
            # sns.barplot(df_tmp.h, df_tmp[cate[page * ppp + i]], orient="h", color="coral", ax=axact)
            axact.barh(df_tmp[cate[page * ppp + i]], df_tmp.h, height=df_tmp.new_w, edgecolor="black")
            axact.set_xlabel("Proportion Target = Y")
            axact.set_title(cate[page * ppp + i] + " (VI:" + str(varimp_cate[cate[page * ppp + i]]) + ")")
            axact.axvline(np.mean(df.target_num), ls="dotted", color="black")
            xlim = axact.get_xlim()
            axact.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
            inset_ax = axact.inset_axes([0, 0, 0.2, 1])
            inset_ax.set_axis_off()
            axact.get_shared_y_axes().join(axact, inset_ax)
            axact.axvline(0, color="black")
            inset_ax.barh(df_tmp[cate[page * ppp + i]], df_tmp.w, color="grey")
            # plt.show()
    # plt.subplots_adjust(wspace=1)
    fig.set_size_inches(w=8, h=6)
    fig.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)
pdf_pages.close()


# Removing variables ----------------------------------------------------------------------------------------------

# Remove leakage variables
cate = np.setdiff1d(cate, ["boat"])
toomany = np.setdiff1d(toomany, ["boat"])

"""
# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot = BoxCore::plot_corr(df[setdiff(cate, paste0("MISS_",miss))], "nomi", cutoff = cutoff_switch)
ggsave(paste0(plotloc,TYPE,"_corr_cate.pdf"), plot, width = 9, height = 9)
if (TYPE %in% c("REGR","MULTICLASS")) {
  plot = BoxCore::plot_corr(df[ paste0("MISS_",miss)], "nomi", cutoff = cutoff_switch)
  ggsave(paste0(plotloc,TYPE,"_corr_cate_MISS.pdf"), plot, width = 9, height = 9)
  cate = setdiff(cate, c("MISS_BsmtFin_SF_2","MISS_BsmtFin_SF_1","MISS_second_Flr_SF","MISS_Misc_Val_LOG_",
                         "MISS_Mas_Vnr_Area","MISS_Garage_Yr_Blt","MISS_Garage_Area","MISS_Total_Bsmt_SF"))
}
"""


"""
# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance
(varimp_cate_fold = filterVarImp(df[cate], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
   .[order(., decreasing = TRUE)] %>% round(2))

# Plot (Hint: one might want to filter just on variable importance with highest importance)
cate_toprint = names(varimp_cate_fold)[varimp_cate_fold >= cutoff_varimp]
plots = map(cate_toprint, ~ BoxCore::plot_distr(df[[.]], df$fold_test, ., "fold_test", varimps = varimp_cate_fold,
                                                colors = c("blue","red")))
ggsave(paste0(plotloc,TYPE,"_distr_cate_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3),
       width = 18, height = 12)
"""


########################################################################################################################
# Prepare final data
########################################################################################################################

# Define final features ----------------------------------------------------------------------------------------

features_notree = np.append(metr, cate)
features_lgbm = np.append(metr, cate + "_ENCODED")
features = np.append(features_notree, toomany + "_ENCODED")
features_binned = np.append(metr_binned, np.setdiff1d(cate, "MISS_" + miss))  # do not need indicators for binned

# Check
np.setdiff1d(features_notree, df.columns.values.tolist())
np.setdiff1d(features, df.columns.values.tolist())
np.setdiff1d(features_binned, df.columns.values.tolist())


# Save image ----------------------------------------------------------------------------------------------------------
del df_orig
dill.dump_session("1_explore.pkl")
