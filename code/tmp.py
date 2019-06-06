
import pandas as pd


def dummy(x):
    x["c"] = 1
    #return x
    return 1

df = pd.DataFrame({"a": [1,2], "b": [1,2]})
df

dummy(df)
df


