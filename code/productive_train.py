# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *

# Specific libraries
from sklearn.pipeline import Pipeline
import xgboost as xgb


# Specific parameters


# ######################################################################################################################
# Fit
# ######################################################################################################################

# --- Read data  ----------------------------------------------------------------------------------------
# ABT
df = pd.read_csv(dataloc + "titanic.csv").iloc[:1000, :]

# Read metadata
df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)\
    .query('status in ["ready","derive"]')

# Define Target
df["target"] = df["survived"]

# Define data to be used for target encoding
np.random.seed(1234)
df["encode_flag"] = np.random.binomial(1, 0.2, len(df))

# Get variable types
nomi = df_meta.loc[df_meta["type"] == "nomi", "variable"].values
ordi = df_meta.loc[df_meta["type"] == "ordi", "variable"].values
cate = np.append(nomi, ordi)
metr = df_meta.loc[df_meta["type"] == "metr", "variable"].values


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
    ("undersample_n", Undersample(n_max_per_level=500))  # undersample
])
df = pipeline_etl.fit_transform(df, df["target"].values, cate_map_nonexist__transform=False)


# --- Fit ----------------------------------------------------------------------------------
# Fit
pipeline_fit = Pipeline([
    ("create_sparse_matrix", CreateSparseMatrix(metr=metr, cate=cate, df_ref=df)),
    ("clf", xgb.XGBClassifier(n_estimators=1100, learning_rate=0.01,
                              max_depth=3, min_child_weight=10,
                              colsample_bytree=0.7, subsample=0.7,
                              gamma=0))
])
fit = pipeline_fit.fit(df, df["target"].values)
# yhat = pipeline_fit.predict_proba(df)  # Test it


# --- Save ----------------------------------------------------------------------------------
with open("productive.pkl", "wb") as f:
    pickle.dump({"pipeline_etl": pipeline_etl,
                 "pipeline_fit": pipeline_fit}, f)
