
# Univariate variable importance
def calc_varimp(df_data, features, target_name="target"):
    # df_data=df; features=metr; target_name="fold"
    varimp = pd.Series()
    for feature_act in features:
        # feature_act=metr[0]
        #IPython.embed()

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
