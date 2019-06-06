# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from init import *

# Specific libraries
from sklearn.pipeline import Pipeline
import xgboost as xgb


# Specific parameters


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data  ----------------------------------------------------------------------------------------
# ABT
df = pd.read_csv(dataloc + "titanic.csv").iloc[:1000, :]
df2 = pd.read_csv(dataloc + "titanic.csv").iloc[1000:, :]

# Read metadata
df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)\
    .query('status in ["ready","derive"]')

# Define Target
df["target"] = df["survived"]

# Define data to be used for target encoding
df["encode_flag"] = np.random.binomial(1, 0.2, len(df))

# Get variable types
nomi = df_meta.loc[df_meta["type"] == "nomi", "variable"].values
ordi = df_meta.loc[df_meta["type"] == "ordi", "variable"].values
cate = np.append(nomi, ordi)
metr = df_meta.loc[df_meta["type"] == "metr", "variable"].values

df_copy = df.copy()

# --- ETL -------------------------------------------------------------------------------------------------
pipeline_etl = Pipeline([
    ("feature_engineering", FeatureEngineeringTitanic(derive_deck=True,
                                                      derive_familysize=True,
                                                      derive_fare_pp=True)),  # feature engineering
    ("cate_convert", Convert(features=cate, convert_to="str")),  # convert cate to "str"
    ("metr_convert", Convert(features=metr, convert_to="float")),  # convert metr to "float"
    ("cate_imp", DfSimpleImputer(features=cate, strategy="constant", fill_value="(Missing)")),  # impute cate with const
    ("cate_map_nonexist", MapNonexisting(features=cate)),  # transform non-existing values: Collect information
    ("cate_enc", TargetEncoding(features=cate, encode_flag_column="encode_flag",
                                target="target")),  # target-encoding
    ("cate_map_toomany", MapToomany(features=cate)),  # reduce "toomany-members" categorical features
    ("metr_imp", DfSimpleImputer(features=metr, strategy="median", verbose=1)),  # impute metr with median
    ("undersmple_n", Undersample(n_max_per_level=500))  # undersample
])
df = pipeline_etl.fit_transform(df, df["target"], cate_map_nonexist__transform=False)
df2 = pipeline_etl.transform(df2)


# --- Fit ----------------------------------------------------------------------------------

# Tuning parameter to use (for xgb)
n_estimators = 1100
learning_rate = 0.01
max_depth = 3
min_child_weight = 10
colsample_bytree = 0.7
subsample = 0.7
gamma = 0

# Fit
pipeline_fit = Pipeline([
    ("create_sparse_matrix", CreateSparseMatrix(metr=metr, cate=cate, df_ref=df)),
    ("clf", xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              max_depth=max_depth, min_child_weight=min_child_weight,
                              colsample_bytree=colsample_bytree, subsample=subsample,
                              gamma=0))
])
X = pipeline_fit.fit(df, df["target"].values)
yhat = pipeline_fit.predict_proba(df)
yhat2 = pipeline_fit.predict_proba(df2)




