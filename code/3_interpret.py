# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries
from sklearn.model_selection import cross_validate

# Main parameter
TARGET_TYPE = "MULTICLASS"

# Specific parameters
n_jobs = 4
labels = None
if TARGET_TYPE == "CLASS":
    metric = "auc"  # metric for peformance comparison
    importance_cut = 99
    topn = 8
    ylim_res = (0, 1)
    color = twocol
elif TARGET_TYPE == "MULTICLASS":
    metric = "auc"  # metric for peformance comparison
    importance_cut = 95
    topn = 15
    ylim_res = (0, 1)
    color = threecol
else:  # "REGR"
    metric = "spear"
    importance_cut = 95
    topn = 15
    ylim_res = (-5e4, 5e4)
    color = None

# Load results from exploration
df = metr_standard = cate_standard = metr_binned = cate_binned = metr_encoded = cate_encoded = target_labels = None
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Features for xgboost
metr = metr_standard
cate = cate_standard
features = np.append([metr, cate])


# ######################################################################################################################
# Prepare
# ######################################################################################################################

# Tuning parameter to use (for xgb) and classifier definition
xgb_param = dict(n_estimators = 1100, learning_rate = 0.01,
                 max_depth = 3, min_child_weight = 10,
                 colsample_bytree = 0.7, subsample = 0.7,
                 gamma = 0,
                 verbosity = 0,
                 n_jobs = n_jobs)
clf = xgb.XGBRegressor(**xgb_param) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(**xgb_param)


# --- Sample data ----------------------------------------------------------------------------------------------------

if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # Training data: Just take data from train fold (take all but n_maxpersample at most)
    df.loc[df["fold"] == "train", "target"].describe()
    under_samp = Undersample(n_max_per_level = 500)
    df_train = under_samp.fit_transform(df.query("fold == 'train'").reset_index(drop = True))
    b_sample = under_samp.b_sample
    b_all = under_samp.b_all
    print(b_sample, b_all)
else:
    df_train = (df.query("fold == 'train'").sample(n = min(df.query("fold == 'train'").shape[0], int(5e3)))
                .reset_index(drop = True))
    b_sample = None
    b_all = None

# Test data
df_test = df.query("fold == 'test'").reset_index(drop = True)  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Combine again
df_traintest = pd.concat([df_train, df_test]).reset_index(drop = True)

# Folds for crossvalidation and check
split_my5fold = TrainTestSep(5, "cv")
for i_train, i_test in split_my5fold.split(df_traintest):
    print("TRAIN-fold:", df_traintest["fold"].iloc[i_train].value_counts())
    print("TEST-fold:", df_traintest["fold"].iloc[i_test].value_counts())
    print("##########")


# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
tr_spm = CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_traintest)
X_train = tr_spm.fit_transform(df_train)
fit = clf.fit(X_train, df_train["target"].values)

# Predict
X_test = tr_spm.transform(df_test)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_test = scale_predictions(fit.predict_proba(X_test), b_sample, b_all)
else:
    yhat_test = fit.predict(X_test)
print(pd.DataFrame(yhat_test).describe())

# Performance
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    print(auc(df_test["target"].values, yhat_test))
else:
    print(spear(df_test["target"].values, yhat_test))

# Plot performance
plot_all_performances(df_test["target"], yhat_test, target_labels = target_labels, target_type = TARGET_TYPE,
                      color = color, ylim = None,
                      pdf = plotloc + TARGET_TYPE + "_performance.pdf")


# --- Check performance for crossvalidated fits ---------------------------------------------------------------------
d_cv = cross_validate(clf, tr_spm.transform(df_traintest), df_traintest["target"],
                      cv = split_my5fold.split(df_traintest),  # special 5fold
                      scoring = d_scoring[TARGET_TYPE],
                      return_estimator = True,
                      n_jobs = 4)
# Performance
print(d_cv["test_" + metric])


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------
# Variable importance (on train data!)
df_varimp_train = calc_varimp_by_permutation(df_train, fit, tr_spm = tr_spm,
                                             target_type = TARGET_TYPE,
                                             b_sample = b_sample, b_all = b_all)

# Top features (importances sum up to 95% of whole sum)
features_top = df_varimp_train.loc[df_varimp_train["importance_cum"] < importance_cut, "feature"].values

# Fit again only on features_top
tr_spm_top = CreateSparseMatrix(metr[np.in1d(metr, features_top)], cate[np.in1d(cate, features_top)],
                                df_ref = df_traintest).fit()
X_train_top = tr_spm_top.transform(df_train)
fit_top = clone(clf).fit(X_train_top, df_train["target"])

# Plot performance
X_test_top = tr_spm_top.transform(df_test)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_top = scale_predictions(fit_top.predict_proba(X_test_top), b_sample, b_all)
    print(auc(df_test["target"].values, yhat_top))
else:
    yhat_top = fit_top.predict(X_test_top)
    print(spear(df_test["target"].values, yhat_top))
plot_all_performances(df_test["target"], yhat_top, target_labels = target_labels, target_type = TARGET_TYPE,
                      color = color, ylim = None,
                      pdf = plotloc + TARGET_TYPE + "_performance_top.pdf")


# ######################################################################################################################
# Diagnosis
# ######################################################################################################################

# ---- Check residuals --------------------------------------------------------------------------------------------

# Residuals
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    df_test["residual"] = 1 - yhat_test[np.arange(len(df_test["target"])), df_test["target"]]  # yhat of true class
else:
    df_test["residual"] = df_test["target"] - yhat_test

df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()

# For non-regr tasks one might want to plot it for each target level (df_test.query("target == 0/1"))
plot_distr(df_test, features,
           target = "residual",
           target_type = "REGR",
           ylim = ylim_res,
           ncol = 3, nrow = 2, w = 18, h = 12,
           pdf = plotloc + TARGET_TYPE + "_diagnosis_residual.pdf")
plt.close(fig = "all")

# Absolute residuals
if TARGET_TYPE == "REGR":
    plot_distr(df = df_test, features = features, target = "abs_residual",
               target_type = "REGR",
               ylim = (0, ylim_res[1]),
               ncol = 3, nrow = 2, w = 18, h = 12,
               pdf = plotloc + TARGET_TYPE + "_diagnosis_absolute_residual.pdf")
plt.close(fig = "all")


# ---- Explain bad predictions ------------------------------------------------------------------------------------

# Get shap for n_worst predicted records
n_worst = 10
df_explain = df_test.sort_values("abs_residual", ascending = False).iloc[:n_worst, :]
yhat_explain = yhat_test[df_explain.index.values]
df_shap = calc_shap(df_explain, fit, tr_spm = tr_spm,
                    target_type = TARGET_TYPE, b_sample = b_sample, b_all = b_all)

# Check
check_shap(df_shap, yhat_explain, target_type = TARGET_TYPE)

# Plot: TODO


# ######################################################################################################################
# Variable Importance
# ######################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------
xgb.plot_importance(fit)


# --- Variable Importance by permuation argument -------------------------------------------------------------------
# Importance for "total" fit (on test data!)
df_varimp = calc_varimp_by_permutation(df_test, fit, tr_spm = tr_spm,
                                       target_type = TARGET_TYPE,
                                       b_sample = b_sample, b_all = b_all)
topn_features = df_varimp["feature"].values[range(topn)]

# Add other information (e.g. special category): category variable is needed -> fill with at least with "dummy"
df_varimp["Category"] = pd.cut(df_varimp["importance"], [-np.inf, 10, 50, np.inf], labels = ["low", "medium", "high"])

# Crossvalidate Importance: ONLY for topn_vars
df_varimp_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(split_my5fold.split(df_traintest)):
    df_tmp = calc_varimp_by_permutation(df_traintest.iloc[i_train, :], d_cv["estimator"][i], tr_spm = tr_spm,
                                        target_type = TARGET_TYPE,
                                        b_sample = b_sample, b_all = b_all,
                                        features = topn_features)
    df_tmp["run"] = i
    df_varimp_cv = df_varimp_cv.append(df_tmp)

# Plot
plot_variable_importance(df_varimp, mask = df_varimp["feature"].isin(topn_features),
                         pdf = plotloc + TARGET_TYPE + "_variable_importance.pdf")
# TODO: add cv lines and errorbars


# --- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------
fig, ax = plt.subplots(1, 1)
sns.barplot("importance_sumnormed", "feature", hue = "fold",
            data = pd.concat([df_varimp_train.assign(fold = "train"), df_varimp.assign(fold = "test")], sort = False))


# ######################################################################################################################
# Partial Dependance
# ######################################################################################################################

# Calc PD
df_pd = calc_partial_dependence(df = df_test, df_ref = df_traintest,
                                fit = fit, tr_spm = tr_spm,
                                target_type = TARGET_TYPE, target_labels = target_labels,
                                b_sample = b_sample, b_all = b_all,
                                features = topn_features)

# Crossvalidate Dependance
df_pd_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(split_my5fold.split(df_traintest)):
    df_tmp = calc_partial_dependence(df = df_traintest.iloc[i_train, :], df_ref = df_traintest,
                                     fit = d_cv["estimator"][i], tr_spm = tr_spm,
                                     target_type = TARGET_TYPE, target_labels = target_labels,
                                     b_sample = b_sample, b_all = b_all,
                                     features = topn_features)
    df_tmp["run"] = i
    df_pd_cv = df_pd_cv.append(df_tmp)

# Plot it
# TODO


# ######################################################################################################################
# Explanations
# ######################################################################################################################

# ---- Explain bad predictions ------------------------------------------------------------------------------------

# Filter data
n_select = 10
i_worst = df_test.sort_values("abs_residual", ascending = False).iloc[:n_select, :].index.values
i_best = df_test.sort_values("abs_residual", ascending = True).iloc[:n_select, :].index.values
i_random = df_test.sample(n = 11).index.values
i_explain = np.unique(np.concatenate([i_worst, i_best, i_random]))
yhat_explain = yhat_test[i_explain]
df_explain = df_test.iloc[i_explain, :].reset_index(drop = True)

# Get shap
df_shap = calc_shap(df_explain, fit, tr_spm = tr_spm,
                    target_type = TARGET_TYPE, b_sample = b_sample, b_all = b_all)

# Check
check_shap(df_shap, yhat_explain, target_type = TARGET_TYPE)

# Plot: TODO

plt.close("all")


# ######################################################################################################################
# Individual dependencies
# ######################################################################################################################

# TODO
