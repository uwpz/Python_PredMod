

from initialize import *

from itertools import product
from sklearn.base import clone
import inspect
from sklearn.model_selection import check_cv

#import sklearn.metrics as foo
getattr(sklearn.metrics, 'roc_auc_score')


'''
estimator = xgb.XGBClassifier() 
X = CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune)
y = df_tune["target"]
param_grid = {"n_estimators": [x for x in range(100, 300, 100)], "learning_rate": [0.01],
              "max_depth": [3,6], "min_child_weight": [5, 10]}
cv = split_my1fold_cv.split(df_tune)
scoring = d_scoring[TARGET_TYPE]
refit = False,
return_train_score = True    
n_jobs = 4
'''

def myGridSearchCV(estimator, X, y, param_grid, cv, scoring, refit=False, return_train_score=False, n_jobs=None):

    # Adapt grid: remove n_estimators
    n_estimators = param_grid.pop("n_estimators")
    df_param_grid = pd.DataFrame(product(*param_grid.values()), columns = param_grid.keys())

    # Materialize generator as this cannot be pickled for parallel
    l_cv = list(cv)

    def run_in_parallel(i):
        # Intialize
        df_results = pd.DataFrame()

        # Get actual parameter set
        d_param = df_param_grid.iloc[[i], :].to_dict(orient = "records")[0]

        for fold, (i_train, i_test) in enumerate(l_cv):

            # Fit only once par parameter set with maximum number of n_estimators
            fit = (clone(estimator).set_params(**d_param,
                                               n_estimators = int(max(n_estimators)))
                   .fit(X[i_train], y[i_train]))

            # Score with all n_estimators
            for ntree_limit in n_estimators:
                yhat_test = fit.predict_proba(X[i_test], ntree_limit = ntree_limit)

                # Do it for training as well
                if return_train_score:
                    yhat_train = fit.predict_proba(X[i_train], ntree_limit = ntree_limit)

                # Get performance metrics
                for scorer in scoring:
                    scorer_value = scoring[scorer]._score_func(y[i_test], yhat_test)
                    df_results = df_results.append(pd.DataFrame(dict(fold_type = "test", fold = fold,
                                                                     scorer = scorer, scorer_value = scorer_value,
                                                                     n_estimators = ntree_limit, **d_param),
                                                                index = [0]))
                    if return_train_score:
                        scorer_value = scoring[scorer]._score_func(y[i_train], yhat_train)
                        df_results = df_results.append(pd.DataFrame(dict(fold_type = "train", fold = fold,
                                                                         scorer = scorer, scorer_value = scorer_value,
                                                                         n_estimators = ntree_limit, **d_param),
                                                                    index = [0]))
        return df_results
    df_results = pd.concat(Parallel(n_jobs = n_jobs, max_nbytes = '100M')(delayed(run_in_parallel)(row)
                                                                          for row in range(len(df_param_grid))))

    # Transform results
    param_names = list(np.append(df_param_grid.columns.values, "n_estimators"))
    df_cv_results = pd.pivot_table(df_results,
                                   values = "scorer_value",
                                   index = param_names,
                                   columns = ["fold_type", "scorer"],
                                   aggfunc = ["mean", "std"],
                                   dropna = False)
    df_cv_results.columns = ['_'.join(x) for x in df_cv_results.columns.values]
    df_cv_results = df_cv_results.reset_index()
    cv_results_ = df_cv_results.to_dict(orient = "list")

    # Refit
    if refit:
        best_param = (df_cv_results[param_names].loc[[df_cv_results["mean_test_" + refit].idxmax()]]
                      .to_dict(orient = "records")[0])
        fit = (clone(estimator).set_params(**best_param).fit(X, y))
    else:
        fit = None

    return {"fit": fit, "cv_results_": cv_results_}


fit = myGridSearchCV(xgb.XGBRegressor(verbosity = 0) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
                     X = (CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune)
                          .fit_transform(df_tune)),
                     y = df_tune["target"],
                     param_grid = {"n_estimators": [x for x in range(100, 300, 100)], "learning_rate": [0.01],
                                   "max_depth": [3, 6], "min_child_weight": [5, 10]},
                     cv = split_my1fold_cv.split(df_tune),
                     scoring = d_scoring[TARGET_TYPE],
                     refit = "auc",
                     return_train_score = True,
                     n_jobs = 4)
print(fit["cv_results_"])
plot_cvresult(cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "max_depth", column_var = "min_child_weight")



from sklearn.model_selection import GridSearchCV

orig = GridSearchCV(xgb.XGBClassifier(verbosity = 0),
                    {"n_estimators": [x for x in range(50, 70, 10)], "learning_rate": [0.01],
                     "max_depth": [3], "min_child_weight": [5]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = "auc",
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = True,
                    # use_warm_start="n_estimators",
                    n_jobs = n_jobs)
orig.decision_function(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune))


class GridSearchCV_xlgb(GridSearchCV):

    def fit(self, X, y=None, **fit_params):
        # Adapt grid: remove n_estimators
        n_estimators = self.param_grid["n_estimators"]
        param_grid = self.param_grid.copy()
        del param_grid["n_estimators"]
        df_param_grid = pd.DataFrame(product(*param_grid.values()), columns = param_grid.keys())

        # Materialize generator as this cannot be pickled for parallel
        self.cv = list(check_cv(self.cv, y).split())

        # Todo
        def run_in_parallel(i):
            # Intialize
            df_results = pd.DataFrame()

            # Get actual parameter set
            d_param = df_param_grid.iloc[[i], :].to_dict(orient = "records")[0]

            for fold, (i_train, i_test) in enumerate(self.cv):

                # Fit only once par parameter set with maximum number of n_estimators
                fit = (clone(self.estimator).set_params(**d_param,
                                                   n_estimators = int(max(n_estimators)))
                       .fit(X[i_train], y[i_train]))

                # Score with all n_estimators
                for ntree_limit in n_estimators:
                    yhat_test = fit.predict_proba(X[i_test], ntree_limit = ntree_limit)

                    # Do it for training as well
                    if self.return_train_score:
                        yhat_train = fit.predict_proba(X[i_train], ntree_limit = ntree_limit)

                    # Get performance metrics
                    for scorer in self.scoring:
                        scorer_value = self.scoring[scorer]._score_func(y[i_test], yhat_test)
                        df_results = df_results.append(pd.DataFrame(dict(fold_type = "test", fold = fold,
                                                                         scorer = scorer, scorer_value = scorer_value,
                                                                         n_estimators = ntree_limit, **d_param),
                                                                    index = [0]))
                        if self.return_train_score:
                            scorer_value = self.scoring[scorer]._score_func(y[i_train], yhat_train)
                            df_results = df_results.append(pd.DataFrame(dict(fold_type = "train", fold = fold,
                                                                             scorer = scorer,
                                                                             scorer_value = scorer_value,
                                                                             n_estimators = ntree_limit, **d_param),
                                                                        index = [0]))
            return df_results

        df_results = pd.concat(Parallel(n_jobs = self.n_jobs,
                                        max_nbytes = '100M')(delayed(run_in_parallel)(row)
                                                             for row in range(len(df_param_grid))))

        # Transform results
        param_names = list(np.append(df_param_grid.columns.values, "n_estimators"))
        df_cv_results = pd.pivot_table(df_results,
                                       values = "scorer_value",
                                       index = param_names,
                                       columns = ["fold_type", "scorer"],
                                       aggfunc = ["mean", "std"],
                                       dropna = False)
        df_cv_results.columns = ['_'.join(x) for x in df_cv_results.columns.values]
        df_cv_results = df_cv_results.reset_index()
        self.cv_results_ = df_cv_results.to_dict(orient = "list")

        # Refit
        if self.refit:
            self.scorer_ = self.scoring
            self.multimetric_ = True
            self.best_index_ = df_cv_results["mean_test_" + self.refit].idxmax()
            self.best_score_ = df_cv_results["mean_test_" + self.refit].loc[self.best_index_]
            self.best_params_ = (df_cv_results[param_names].loc[[self.best_index_]]
                                 .to_dict(orient = "records")[0])
            self.best_estimator_ = (clone(self.estimator).set_params(**self.best_params_).fit(X, y))

        return self


custom = GridSearchCV_xlgb(xgb.XGBClassifier(verbosity = 0),
                           {"n_estimators": [x for x in range(50, 70, 10)], "learning_rate": [0.01],
                            "max_depth": [3], "min_child_weight": [5]},
                           cv = split_my1fold_cv.split(df_tune),
                           refit = "auc",
                           scoring = d_scoring[TARGET_TYPE],
                           return_train_score = True,
                           n_jobs = n_jobs)
a = custom.fit(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune),
     df_tune["target"])

a.decision_function(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune))

