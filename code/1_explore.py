
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from scipy.stats.mstats import winsorize

# Main parameter
TARGET_TYPE = "REGR"

# Specific parameters
if TARGET_TYPE == "CLASS":
    ylim = None
    cutoff_corr = 0.1
    cutoff_varimp = 0.52
    color = twocol
if TARGET_TYPE == "REGR":
    ylim = (0, 250e3)
    cutoff_corr = 0.8
    cutoff_varimp = 0.52
    color = None


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data ------------------------------------------------------------------------------------------------------
# noinspection PyUnresolvedReferences
if TARGET_TYPE == "CLASS":
    df_orig = pd.read_csv(dataloc + "titanic.csv")
if TARGET_TYPE == "REGR":
    df_orig = pd.read_csv(dataloc + "AmesHousing.txt", delimiter="\t")
    df_orig.columns = df_orig.columns.str.replace(" ","_")
df_orig.describe()
df_orig.describe(include=["object"])

"""
# Check some stuff
df_values = create_values_df(df_orig, 10)
print(df_values)
if TARGET_TYPE == "CLASS":
    df_orig["survived"].value_counts() / df_orig.shape[0]
if TARGET_TYPE == "REGR":
    fig, ax = plt.subplots(1, 2)
    df_orig["SalePrice"].plot.hist(bins=20, ax = ax[0])
    np.log(df_orig["SalePrice"]).plot.hist(bins=20, ax = ax[1]);
"""

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) -----------------------------------------------------------------------------
if TARGET_TYPE == "CLASS":
    df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header=1)
if TARGET_TYPE == "REGR":
    df_meta = pd.read_excel(dataloc + "datamodel_AmesHousing.xlsx", header=1)

# Check
print(np.setdiff1d(df.columns.values, df_meta["variable"].values, assume_unique=True))
print(np.setdiff1d(df_meta.loc[df_meta["category"] == "orig", "variable"].values, df.columns.values,
                   assume_unique=True))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready", "derive"])]


# --- Feature engineering -----------------------------------------------------------------------------------------
if TARGET_TYPE == "CLASS":
    df["deck"] = df["cabin"].str[:1]
    df["familysize"] = df["sibsp"] + df["parch"] + 1
    df["fare_pp"] = df["fare"] / df.groupby("ticket")["fare"].transform("count")
    df[["deck", "familysize", "fare_pp"]].describe(include="all")
if TARGET_TYPE == "REGR":
    pass # number of rooms, sqm_per_room, ...

# Check
print(np.setdiff1d(df_meta["variable"].values, df.columns.values, assume_unique=True))


# --- Define target and train/test-fold ----------------------------------------------------------------------------
# Target
if TARGET_TYPE == "CLASS":
    df["target"] = df["survived"]
if TARGET_TYPE == "REGR":
    df["target"] = df["SalePrice"]
df["target"].describe()

# Train/Test fold: usually split by time
np.random.seed(123)
# noinspection PyTypeChecker
df["fold"] = np.random.permutation(pd.qcut(np.arange(len(df)), q=[0, 0.1, 0.8, 1], labels=["util", "train", "test"]))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].map({"train": 0, "util": 0, "test": 1})

# Define the id
df["id"] = np.arange(len(df)) + 1

# Define data to be used for target encoding
df["encode_flag"] = np.random.binomial(1, 0.2, len(df))


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------
metr = df_meta_sub.loc[df_meta_sub.type == "metr", "variable"].values
df = Convert(features=metr, convert_to="float").fit_transform(df)
df[metr].describe()


# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
metr_binned = metr + "_BINNED_"
df[metr_binned] = df[metr].apply(lambda x: pd.qcut(x, 10, duplicates="drop").astype("str"))

# Convert missings to own level ("(Missing)")
df[metr_binned] = df[metr_binned].replace("nan", np.nan).fillna("(Missing)")
print(create_values_df(df[metr_binned], 11))

# Remove binned variables with just 1 bin
onebin = metr_binned[df[metr_binned].nunique() == 1]
metr_binned = np.setdiff1d(metr_binned, onebin, assume_unique=True)


# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------
# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
misspct.sort_values(ascending=False)  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
print(remove)
metr = np.setdiff1d(metr, remove, assume_unique=True)  # adapt metadata
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED_", assume_unique=True)  # keep "binned" version in sync

# Check for outliers and skewness
plot_distr(df.query("fold != 'util'"), metr, target_type=TARGET_TYPE, color=color, ylim=ylim,
           ncol=3, nrow=2, w=12, h=8, pdf=plotloc + TARGET_TYPE + "_distr_metr.pdf")

# Winsorize
df[metr] = df[metr].apply(lambda x: winsorize(x, (0.01, 0.01)))  # hint: plot again before deciding for log-trafo

# Log-Transform
if TARGET_TYPE == "CLASS":
    tolog = np.array(["fare"], dtype="object")
if TARGET_TYPE == "REGR":
    tolog = np.array(["Lot_Area"], dtype="object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - max(0, np.min(x)) + 1))
np.place(metr, np.isin(metr, tolog), tolog + "_LOG_")  # adapt metadata (keep order)


# --- Final variable information ------------------------------------------------------------------------------------
# Univariate variable importance
varimp_metr = calc_imp(df.query("fold != 'util'"), metr, target_type=TARGET_TYPE)
print(varimp_metr)
varimp_metr_binned = calc_imp(df.query("fold != 'util'"), metr_binned, target_type=TARGET_TYPE)
print(varimp_metr_binned)

# Plot 
plot_distr(df.query("fold != 'util'"), metr, varimp=varimp_metr, target_type=TARGET_TYPE, color=twocol, ylim=ylim,
           ncol=3, nrow=2, w=12, h=8, pdf=plotloc + TARGET_TYPE + "_distr_metr_final.pdf")


# --- Removing variables -------------------------------------------------------------------------------------------
# Remove leakage features
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove, assume_unique=True)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED", assume_unique=True)  # keep "binned" version in sync

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df, metr, cutoff=cutoff_corr, pdf=plotloc + TARGET_TYPE + "_corr_metr.pdf")
remove = np.array(["xxx", "xxx"], dtype="object")
metr = np.setdiff1d(metr, remove, assume_unique=True)
metr_binned = np.setdiff1d(metr_binned, remove + "_BINNED", assume_unique=True)  # keep "binned" version in sync


# --- Time/fold depedency --------------------------------------------------------------------------------------------
# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_fold = calc_imp(df.query("fold != 'namethisutil'"), metr, "fold_num")

# Plot: only variables with with highest importance
metr_toprint = varimp_metr_fold[varimp_metr_fold > cutoff_varimp].index.values
plot_distr(df.query("fold != 'namethisutil'"), metr_toprint, "fold_num", varimp=varimp_metr_fold, target_type="CLASS",
           ncol=2, nrow=2, w=12, h=8, pdf=plotloc + TARGET_TYPE + "_distr_metr_folddep.pdf")


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------
miss = metr[df[metr].isnull().any().values]  # alternative: [x for x in metr if df[x].isnull().any()]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
df = DfSimpleImputer(features=miss, strategy="median").fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------
# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["nomi", "ordi"]), "variable"].values
df[cate] = df[cate].astype("str").replace("nan", np.nan)
df[cate].describe()

# Merge categorical variable (keep order)
cate = np.append(cate, ["MISS_" + miss])


# --- Handling factor values ----------------------------------------------------------------------------------------
# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 10
levinfo = df[cate].apply(lambda x: x.unique().size).sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = np.setdiff1d(toomany, ["xxx", "xxx"], assume_unique=True)  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features=cate, encode_flag_column="encode_flag", target="target").fit_transform(df)

# Convert toomany features: lump levels and map missings to own level
df = MapToomany(features=toomany, n_top=10).fit_transform(df)

# Univariate variable importance
varimp_cate = calc_imp(df.query("fold != 'namethisutil'"), cate, target_type=TARGET_TYPE)
print(varimp_cate)

# Check
plot_distr(df.query("fold != 'namethisutil'"), cate, varimp=varimp_cate, target_type=TARGET_TYPE,
           color=color, ylim=ylim,
           nrow=2, ncol=3, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_cate.pdf")


# --- Removing variables ---------------------------------------------------------------------------------------------
# Remove leakage variables
cate = np.setdiff1d(cate, ["xxx"], assume_unique=True)
toomany = np.setdiff1d(toomany, ["xxx"], assume_unique=True)

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df, cate, cutoff=cutoff_corr, n_cluster=5, pdf=plotloc + TARGET_TYPE + "_corr_cate.pdf")


# --- Time/fold depedency --------------------------------------------------------------------------------------------
# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_fold = calc_imp(df.query("fold != 'namethisutil'"), cate, "fold_num")

# Plot: only variables with with highest importance
cate_toprint = varimp_cate_fold[varimp_cate_fold > cutoff_varimp].index.values
plot_distr(df.query("fold != 'namethisutil'"), cate_toprint, "fold_num", varimp=varimp_cate_fold, target_type="CLASS",
           ncol=2, nrow=2, w=12, h=8, pdf=plotloc + TARGET_TYPE + "_distr_cate_folddep.pdf")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features ----------------------------------------------------------------------------------------
features = np.concatenate([metr, cate, toomany + "_ENCODED"])
features_binned = np.concatenate([metr_binned, np.setdiff1d(cate, "MISS_" + miss, assume_unique=True),
                                  toomany + "_ENCODED"])  # do not need indicators for binned
features_lgbm = np.append(metr, cate + "_ENCODED")

# Check
np.setdiff1d(features, df.columns.values.tolist(), assume_unique=True)
np.setdiff1d(features_binned, df.columns.values.tolist(), assume_unique=True)
np.setdiff1d(features_lgbm, df.columns.values.tolist(), assume_unique=True)


# --- Remove burned data ----------------------------------------------------------------------------------------
df = df.query("fold != 'namethisutil'")


# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(TARGET_TYPE + "_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "metr": metr,
                 "cate": cate,
                 "features": features,
                 "features_binned": features_binned,
                 "features_lgbm": features_lgbm},
                file)
