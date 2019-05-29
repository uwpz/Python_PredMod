df_fitres = pd.DataFrame.from_dict(fit.cv_results_)
df_fitres["param_min_child_weight__learning_rate"] = (df_fitres.param_min_child_weight.astype("str") + "_" +
                                                      df_fitres.param_learning_rate.astype("str"))
sns.catplot(kind="point",
            data=df_fitres,
            x="param_n_estimators", y="mean_test_score", hue="param_min_child_weight__learning_rate",
            col="param_max_depth",
            palette=["C0", "C0", "C1", "C1"], markers=["o", "x", "o", "x"], linestyles=["-", ":", "-", ":"],
            legend_out=False)
df_fitres.pivot_table(values=["mean_test_score"],
                     index="param_n_estimators",
                     columns=["param_min_child_weight__learning_rate","param_max_depth"]) \
    .plot(marker="o")
plt.close()
# Score

fit = RandomForestClassifier(n_estimators=50, max_features=3)\
    .fit(create_sparse_matrix(df_tune, metr, cate), df_tune["target"])
yhat = fit.predict_proba(create_sparse_matrix(df_tune, metr, cate))[:, 1]
roc_auc_score(df_tune["target"], yhat)
fpr, tpr, cutoff = roc_curve(df_tune["target"], yhat)
cross_val_score(fit,
                create_sparse_matrix(df_tune, metr, cate), df_tune["target"],
                cv=5, scoring=metric, n_jobs=5)


from plotnine import *

df = {"dates": [1, 2, 3, 4, 5, 6], "amount": [21, 22, 18, 19, 25, 15]}
df = pd.DataFrame(df)
plt.ioff()
plot = ggplot(aes(x="dates", y="amount"), data=df) + xlab("Dates") + ylab("Amount")  # Create the base plot and axes
plot


factors = ["param_min_child_weight", "param_learning_rate", "param_max_depth"]
df_fitres[factors] = df_fitres[factors].astype("str")
(ggplot(df_fitres, aes(x="param_n_estimators",
                       y="mean_test_score",
                       colour="param_min_child_weight"))
  + geom_line(aes(linetype="param_learning_rate"))
  + geom_point(aes(shape="param_learning_rate"))
  + facet_grid(". ~ param_max_depth"))


# plotnine cannot plot several plots on one page
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


#%matplotlib
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
plt.close("all")
plt.plot(x, np.sin(x), ".")
plt.plot(x, np.cos(x))
fig.set_size_inches(w = 6/2.54, h = 8/2.54)
fig.savefig("firstfig.pdf")

plt.close("all")
fig, ax = plt.subplots()
ax = df["age"].hist()
df.age.plot.kde(ax = ax, secondary_y = True)
#plt.show()
#plt.draw()


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_train = encoder.fit_transform(df_train[nomi])


import pandas as pd
a = pd.DataFrame({"x" : pd.Series(["a","b","c"]), "y" : pd.Series([0,0,0])})
a["x"] = a["x"].astype("category")
a.dtypes
b = pd.get_dummies(a)

c = a.loc[a.x!="c",:]
c.dtypes
pd.get_dummies(c)


import matplotlib.pyplot as plt

p_df = pd.DataFrame({"class": [1,1,2,2,1], "a": [2,3,2,3,2]})
fig, ax = plt.subplots(figsize=(8,6))
p_df.groupby('class').plot(kind='line')
p_df.groupby("class").plot.line()
