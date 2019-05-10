# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:38:09 2017

@author: Uwe
"""

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
