# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm
plt.ion(); matplotlib.use('TkAgg')


# Specific libraries
from sklearn.linear_model import ElasticNet

# Main parameter
TARGET_TYPE = "REGR"

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
features = np.concatenate([metr, cate])

# Scale "metr" features for DL (Tree-based are not influenced by this Trafo)
#df[metr] = (df[metr] - df[metr].min()) / (df[metr].max() - df[metr].min())
df[metr] = (df[metr] - df[metr].mean()) / (df[metr].std())
df[metr].describe()
#df["target"] = (df["target"] - df["target"].mean()) / (df["target"].std())




# ######################################################################################################################
# Prepare
# ######################################################################################################################

# Lasso / Elastic Net
split_5fold = KFold(5, shuffle=True, random_state=42)
fit = (GridSearchCV(ElasticNet(warm_start = True, fit_intercept = True) ,  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-2, 10, 2)],
                     "l1_ratio": [1]},
                    cv = split_5fold.split(df),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = True,
                    n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr, cate = cate, df_ref = df).fit_transform(df),
            df["target"]))
plot_cvresult(fit.cv_results_, metric = metric, x_var = "alpha", color_var = "l1_ratio")
a = pd.DataFrame(fit.cv_results_)


# Fit
elasticnet_param = dict(alpha = 2**7, l1_ratio = 1, fit_intercept = True)
clf = ElasticNet(**elasticnet_param)
tr_spm = CreateSparseMatrix(metr = metr, cate = cate, df_ref = df)
X_train = tr_spm.fit_transform(df)
fit = clf.fit(X_train, df["target"].values)
fit.coef_
fit.intercept_



# Predict
X_test = tr_spm.transform(df)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_test = scale_predictions(fit.predict_proba(X_test), b_sample, b_all)
else:
    yhat_test = fit.predict(X_test)
print(pd.DataFrame(yhat_test).describe())

# Performance
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    print(auc(df["target"].values, yhat_test))
else:
    print(spear(df["target"].values, yhat_test))



# ######################################################################################################################
# Betas
# ######################################################################################################################

# --- Calc cv values ---------------------------------------------------------------------------------------------------

# Calc betas
def calc_beta_cv(df, clf, target, metr, cate, random_seed=999, n_fold=5, n_rep=10):

    # Define split
    split = RepeatedKFold(n_fold, n_repeats=n_rep, random_state=random_seed).split(df, df[target])

    # Loop
    df_beta_cv = pd.DataFrame()
    for i_rep in (np.arange(n_rep)+1):
        for i_fold in (np.arange(n_fold)+1):
            #i_rep=3; i_fold=1;

            print("i_rep:", i_rep)
            print("i_fold:", i_fold)

            # Create fold
            i_train, i_test = next(split)

            # Fit train
            tr_spm = CreateSparseMatrix(metr=metr, cate=cate, df_ref=df)
            fit_train = clone(clf).fit(tr_spm.fit_transform(df.iloc[i_train]), df[target].iloc[i_train])
            df_beta = tr_spm.df_map
            df_beta["variable"] = np.where(~df_beta["value"].isnull(),
                                           df_beta["variable"] + " = " + df_beta["value"],
                                           df_beta["variable"])
            df_beta = df_beta.assign(beta = fit_train.coef_, rep = i_rep, fold = i_fold)
            df_beta_cv = pd.concat([df_beta_cv, df_beta], ignore_index=True)

    return df_beta_cv

# 5fold-10timesrepeated
df_beta_cv = (calc_beta_cv(df, clf=clf, target="target", metr=metr, cate=cate, random_seed=42, n_fold=5, n_rep=3)
              .query("beta > 0").reset_index(drop = True))


# --- Beta Cv selection plot --------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,2)
order = df_beta_cv["variable"].value_counts().index
#order = np.sort(df_varimp_cv["feature"].unique())

# Selection plot
ax_act = ax.flat[0]
sns.countplot(y="variable", data=df_beta_cv, order=order, ax=ax_act)
inset_ax = ax_act.inset_axes([0.6, 0.05, 0.3, 0.3])
ax_act.set_title("Number of selections")
inset_ax.boxplot(df_beta_cv.groupby(["rep", "fold"])["variable"].count().reset_index()["variable"], showmeans=True)
inset_ax.get_xaxis().set_visible(False)
inset_ax.set_title("Features selected per CV-fold")

# VI distribution
ax_act = ax.flat[1]
sns.boxplot(y="variable", x="beta", orient="h", data=df_beta_cv, order=order, ax=ax_act)
ax_act.set_title("Betas over repetitions")

fig.set_size_inches(w=18, h=12)
fig.tight_layout()
fig.savefig(plotloc + "variable_importance_cv.pdf")


# ######################################################################################################################
# Performances
# ######################################################################################################################

# --- Calc cv values ---------------------------------------------------------------------------------------------------



# Calc CV yhat
def calc_cv(df, df_ref, clf, target, metr, cate,  id, target_type = "CLASS", vicum_cutoff=99, random_seed=999, n_fold=5, n_rep=10, n_jobs=4):
#df_ref=df; vicum_cutoff=99; random_seed=42; target="target"; id="PID"; target_type = TARGET_TYPEn_fold=5; n_rep=10; n_jobs=4;

    # Define split
    split = RepeatedKFold(n_fold, n_repeats=n_rep, random_state=random_seed).split(df, df[target])

    # Loop
    df_varimp = pd.DataFrame()
    df_yhat = pd.DataFrame()
    for i_rep in (np.arange(n_rep)+1):
        for i_fold in (np.arange(n_fold)+1):
            #i_rep=3; i_fold=1;
            #for i in range(10):
            #    i_train,i_test=next(split)
            print("i_rep:", i_rep)
            print("i_fold:", i_fold)

            # Create fold
            i_train, i_test = next(split)

            # Fit and scorefull model
            fit_full = clone(clf).fit(
                CreateSparseMatrix(metr=metr, cate=cate, df_ref=df).fit_transform(df.iloc[i_train]),
                df[target].iloc[i_train])
            if target_type in ["CLASS", "MULTICLASS"]:
                yhat_full = fit_full.predict_proba(CreateSparseMatrix(metr=metr, cate=cate, df_ref=df).
                                                   fit_transform(df.iloc[i_test]))
            else:
                yhat_full = fit_full.predict(CreateSparseMatrix(metr = metr, cate = cate, df_ref = df).
                                                   fit_transform(df.iloc[i_test]))
            df_yhat_full = pd.concat([df[id].iloc[i_test].reset_index(drop=True),
                                      pd.DataFrame(yhat_full)], axis=1) \
                .assign(type="full", rep=i_rep, fold=i_fold)
            df_yhat = pd.concat([df_yhat, df_yhat_full], ignore_index=True)

            # VI full model (train data)
            df_vi = calc_varimp_by_permutation(df.iloc[i_train].reset_index(drop=True), df_ref=df, fit=fit_full,
                                               target="target", metr=metr, cate=cate,
                                               target_type = target_type,
                                               random_seed=random_seed, n_jobs=n_jobs)
            if sum(df_vi["importance_cum"] < vicum_cutoff) == 0:
                df_vi = df_vi.iloc[[0], :].reset_index(drop=True)
            else:
                df_vi = df_vi.loc[df_vi["importance_cum"] < vicum_cutoff].reset_index(drop=True)
            features_vi = df_vi["feature"].values
            metr_vi = metr[np.isin(metr, features_vi)]
            cate_vi = cate[np.isin(cate, features_vi)]
            df_varimp = pd.concat([df_varimp,
                                   df_vi.assign(type="full", rep=i_rep, fold=i_fold)])

            # Fit and score reduced model
            fit_vi = clone(clf).fit(CreateSparseMatrix(metr=metr_vi, cate=cate_vi, df_ref=df)\
                                    .fit_transform(df.iloc[i_train]),
                                    df[target].iloc[i_train])
            if target_type in ["CLASS", "MULTICLASS"]:
                yhat_vi = fit_vi.predict_proba(CreateSparseMatrix(metr = metr_vi, cate = cate_vi, df_ref = df).
                                               fit_transform(df.iloc[i_test]))
            else:
                yhat_vi = fit_vi.predict(CreateSparseMatrix(metr=metr_vi, cate=cate_vi, df_ref=df).
                                         fit_transform(df.iloc[i_test]))
            df_yhat_vi = pd.concat([df[id].iloc[i_test].reset_index(drop=True),
                                    pd.DataFrame(yhat_vi)], axis=1) \
                .assign(type="vi", rep=i_rep, fold=i_fold)
            df_yhat = pd.concat([df_yhat, df_yhat_vi], ignore_index=True)

    return df_yhat, df_varimp


# 5fold-10timesrepeated
df_yhat_cv, df_varimp_cv = calc_cv(df, df_ref=df, clf=clf,
                                   target="target", metr=metr, cate=cate, id="PID", target_type = TARGET_TYPE,
                                   vicum_cutoff = 99, random_seed=42,
                                   n_fold=5, n_rep=3, n_jobs=4)



# --- VI Cv selection plot --------------------------------------------------------------------------------------------

fig, ax=plt.subplots(1,2)
order = df_varimp_cv["feature"].value_counts().index
#order = np.sort(df_varimp_cv["feature"].unique())

# Selection plot
ax_act = ax.flat[0]
sns.countplot(y="feature", data=df_varimp_cv, order=order, ax=ax_act)
inset_ax = ax_act.inset_axes([0.6, 0.05, 0.3, 0.3])
ax_act.set_title("Number of selections")
inset_ax.boxplot(df_varimp_cv.groupby(["rep", "fold"])["feature"].count().reset_index()["feature"], showmeans=True)
inset_ax.get_xaxis().set_visible(False)
inset_ax.set_title("Features selected per CV-fold")

# VI distribution
ax_act = ax.flat[1]
sns.boxplot(y="feature", x="importance", orient="h", data=df_varimp_cv, order=order, ax=ax_act)
ax_act.set_title("Variable importance over repetitions")

fig.set_size_inches(w=18, h=12)
fig.tight_layout()
fig.savefig(plotloc + "variable_importance_cv.pdf")