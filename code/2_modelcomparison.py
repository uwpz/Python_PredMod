
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *

# Specific libraries
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, learning_curve
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

# Main parameter
TARGET_TYPE = "REGR"

# Specific parameters
n_jobs = 4
if TARGET_TYPE == "CLASS":
    scoring = {"auc": make_scorer(auc, greater_is_better=True),
               "acc": make_scorer(acc, greater_is_better=True)}
    metric = "auc" # metric for peformance comparison
if TARGET_TYPE == "REGR":
    scoring = {"spear": make_scorer(spearman_loss_func, greater_is_better=True),
               "rmse": make_scorer(rmse, greater_is_better=False)}
    metric = "spear"

# Load results from exploration
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ----------------------------------------------------------------------------------------------------
if TARGET_TYPE == "CLASS":
    # Undersample only training data (take all but n_maxpersample at most)
    under_samp = Undersample(n_max_per_level=500)
    df_tmp = under_samp.fit_transform(df)
    b_all = under_samp.b_all
    b_sample = under_samp.b_sample
    print(b_sample, b_all)
    # df_tmp, b_sample, b_all = undersample_n(df[df["fold"] == "train"], 500)
    df_tune = pd.concat([df_tmp, df.loc[df["fold"] == "test"]], sort=False).reset_index()
    df_tune.groupby("fold")["target"].describe()

if TARGET_TYPE == "REGR":
    df_tune = df.sample(n=min(df.shape[0], int(5e3))).reset_index()


# --- Define some splits -------------------------------------------------------------------------------------------
# split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_my1fold_cv = TrainTestSep(1)
# split_5fold = KFold(5, shuffle=False, random_state=42)
split_my5fold_cv = TrainTestSep(5)
split_my5fold_boot = TrainTestSep(5, "bootstrap")
'''
df_tune["fold"].value_counts()
split_my5fold = TrainTestSep(n_splits=5, sample_type="cv")
iter_split = split_my5fold.split(df_tune)
i_train, i_test = next(iter_split)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------

# Lasso / Elastic Net
fit = GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                   [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                     "l1_ratio": [0, 0.2, 0.5, 0.8, 1]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   n_jobs=n_jobs)\
    .fit(CreateSparseMatrix(metr=metr_binned, cate=cate_binned, df_ref=df_tune).fit_transform(df_tune),
         df_tune["target"])
plot_cvresult(fit.cv_results_, metric=metric, x_var="alpha", color_var="l1_ratio")
# -> keep l1_ratio=1 to have a full Lasso


# Random Forest
fit = GridSearchCV(RandomForestRegressor() if TARGET_TYPE == "REGR" else RandomForestClassifier(),
                   [{"n_estimators": [10, 20],
                     "max_features": [x for x in range(1, len(metr_standard)+len(cate_standard), 5)]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   # use_warm_start=["n_estimators"],
                   n_jobs=n_jobs)\
    .fit(CreateSparseMatrix(metr=metr_standard, cate=cate_standard, df_ref=df_tune).fit_transform(df_tune),
         df_tune["target"])
plot_cvresult(fit.cv_results_, metric=metric, x_var="n_estimators", color_var="max_features")

# -> keep around the recommended values: max_features = floor(sqrt(length(features)))


# XGBoost
fit = GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                   [{"n_estimators": [x for x in range(100,3100,500)], "learning_rate": [0.01],
                     "max_depth": [3,6], "min_child_weight": [5,10]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   # use_warm_start="n_estimators",
                   n_jobs=n_jobs) \
    .fit(CreateSparseMatrix(metr=metr_standard, cate=cate_standard, df_ref=df_tune).fit_transform(df_tune),
         df_tune["target"])
# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10
df_fitres = pd.DataFrame(fit.cv_results_)
df_fitres.mean_fit_time.values.mean()
plot_cvresult(fit.cv_results_, metric=metric,
              x_var="n_estimators", color_var="max_depth", column_var="min_child_weight")


# LightGBM
fit = GridSearchCV(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else lgbm.LGBMClassifier(),
                   [{"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                     "num_leaves": [8,32,64], "min_child_samples": [5,10]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   n_jobs=n_jobs) \
    .fit(df_tune[metr_encoded], df_tune["target"],
         categorical_feature=[x for x in metr_encoded.tolist() if "_ENCODED" in x])
plot_cvresult(fit.cv_results_, metric=metric,
              x_var="n_estimators", color_var="num_leaves", column_var="min_child_samples")


# ######################################################################################################################
# Evaluate generalization gap
# ######################################################################################################################

# Sample data (usually undersample training data)
df_gengap = df_tune.copy()

# Tune grid to loop over
param_grid = [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
               "max_depth": [3, 6], "min_child_weight": [5, 10],
               "colsample_bytree": [0.7], "subsample": [0.7],
               "gamma": [10]}]

# Calc generalization gap
fit = GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                   param_grid,
                   cv=split_my1fold_cv.split(df_gengap),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   n_jobs=n_jobs) \
    .fit(CreateSparseMatrix(metr=metr_standard, cate=cate_standard, df_ref=df_gengap).fit_transform(df_gengap),
         df_gengap["target"])
plot_gengap(fit.cv_results_, metric = metric,
            x_var="n_estimators", color_var="max_depth", column_var="min_child_weight", row_var="gamma",
            pdf=plotloc + TARGET_TYPE + "_xgboost_gengap.pdf")


# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
      estimator=GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                             [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                               "l1_ratio": [1]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=metric,
                             scoring=scoring,
                             return_train_score=False,
                             n_jobs=n_jobs),
      X=CreateSparseMatrix(metr=metr_binned, cate=cate_binned, df_ref=df_modelcomp).fit_transform(df_modelcomp),
      y=df_modelcomp["target"],
      cv=split_my5fold_cv.split(df_modelcomp),
      return_train_score=False,
      n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
      estimator=GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                             [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                               "max_depth": [6], "min_child_weight": [10]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=metric,
                             scoring=scoring,
                             return_train_score=False,
                             n_jobs=n_jobs),
      X=CreateSparseMatrix(metr=metr_standard, cate=cate_standard, df_ref=df_modelcomp).fit_transform(df_modelcomp),
      y=df_modelcomp["target"],
      cv=split_my5fold_cv.split(df_modelcomp),
      return_train_score=False,
      n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)


# --- Plot model comparison ------------------------------------------------------------------------------

plot_modelcomp(df_modelcomp_result.rename(columns = {"index": "run", "test_score": metric}), scorevar = metric,
               pdf=plotloc + TARGET_TYPE + "_model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################


# Basic data sampling
df_lc = df_tune.copy()

# Calc learning curve
n_train, score_train, score_test = learning_curve(
      estimator=GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                             [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                               "max_depth": [3], "min_child_weight": [10]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=metric,
                             scoring=scoring,
                             return_train_score=False,
                             n_jobs=4),
      X=CreateSparseMatrix(metr=metr_standard, cate=cate_standard, df_ref=df_lc).fit_transform(df_lc),
      y=df_lc["target"],
      train_sizes=np.append(np.linspace(0.05, 0.1, 5), np.linspace(0.2, 1, 5)),
      cv=split_my1fold_cv.split(df_lc),
      n_jobs=4)

# Plot it
plot_learning_curve(n_train, score_train, score_test,
                    pdf=plotloc + TARGET_TYPE + "_learningCurve.pdf")

plt.close("all")
