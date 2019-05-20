

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
