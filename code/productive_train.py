
# ######################################################################################################################
#  Libraries
# ######################################################################################################################

# --- Load general libraries  -------------------------------------------------------------------------------------
# Data
import numpy as np
import pandas as pd
#from dill import (load_session, dump_session)

# Util
import pdb  # pdb.set_trace()  #quit with "q", next line with "n", continue with "c"
from os import getcwd

# My
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from myfunc import *


# --- Load specific libraries  -------------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer


# --- Load results, run 0_init  -----------------------------------------------
exec(open("./code/0_init.py").read())


# ######################################################################################################################
# Initialize
# ######################################################################################################################

# Adapt some default parameter differing per target types (NULL results in default value usage)
# cutoff_switch = switch(TARGET_TYPE, "CLASS" = 0.1, "REGR" = 0.8, "MULTICLASS" = 0.8)  # adapt
# ylim_switch = switch(TARGET_TYPE, "CLASS" = NULL, "REGR" = c(0, 250e3), "MULTICLASS" = NULL)  # adapt REGR opt
cutoff_corr = 0.1
cutoff_varimp = 0.52


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data and transform ----------------------------------------------------------------------------------------
# ABT
df = pd.read_csv(dataloc + "titanic.csv").iloc[0:1000, :]

#  Feature engineering
df["deck"] = df["cabin"].str[:1]
df["familysize"] = df["sibsp"] + df["parch"] + 1
df["fare_pp"] = df.groupby("ticket")["fare"].transform("mean")

# Read metadata
df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)\
    .query('status in ["ready","derive"]')

# Define Target
df["target"] = df["survived"]


# --- Adapt categorical variables ----------------------------------------------------------------------------------
nomi = df_meta.loc[df_meta["type"] == "nomi", "variable"].values
ordi = df_meta.loc[df_meta["type"] == "ordi", "variable"].values
cate = np.append(nomi, ordi)

# Convert to string
df[cate].describe(include="all")
df[cate] = df[cate].astype("str").replace("nan",np.nan)

# Impute "(Missing)"
imp_const = DfSimpleImputer(features=cate, strategy="constant", fill_value="(Missing)")
df = imp_const.fit_transform(df)

# Transform non-existing values: Collect information
map_nonexist = MapNonexisting(features=nomi)
df = map_nonexist.fit_transform(df, transform=False)

# Create target-encoded features, i.e. numeric representation
enc_cate = TargetEncoding(features=cate, df4encoding=df, target="target")
df = enc_cate.fit_transform(df)

# Reduce "toomany-members" categorical features
map_toomany = MapToomany(features=cate)
df = map_toomany.fit_transform(df)

from sklearn.pipeline import Pipeline
pipeline1 = Pipeline([
    ("cate_imp", DfSimpleImputer(features=cate, strategy="constant", fill_value="(Missing)")),
    ("cate_map_nonexist", MapNonexisting(features=nomi)),
    ("cate_enc", TargetEncoding(features=cate, df4encoding=df, target="target")),
    ("cate_map_toomany", MapToomany(features=cate)),
    ("metr_imp", DfSimpleImputer(features=metr, strategy="median"))
])
tmp1 = pipeline1.fit_transform(df, cate_map_nonexist__transform=False)



# # --- Adapt metric variables -------------------------------------------------------------------------------------
# metr = df_meta.loc[df_meta["type"] == "metr", "variable"].values
#
# # Impute median
# imp_median = DfSimpleImputer(features=metr, strategy="median")
# df = imp_median.fit_transform(df)



