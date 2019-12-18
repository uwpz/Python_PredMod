# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *

# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries


# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters (CLASS is default)
ylim = None
cutoff_corr = 0.85
cutoff_varimp = 0.52
color = twocol[::-1]
min_width = 0


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data ------------------------------------------------------------------------------------------------------
colnames = pd.read_table(dataloc + "census_colnames.txt", header = None, delimiter = "\t").loc[:,0].values
df_orig = pd.read_csv(dataloc + "census-income.data", header = None, names=colnames)

# Quick check
df_orig.describe()
df_orig.describe(include = ["object"])
df_values = create_values_df(df_orig, 10)
df_orig["income"].value_counts() / len(df_orig)

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) -----------------------------------------------------------------------------

df_meta = pd.read_excel(dataloc + "datamodel_census.xlsx", header = 1)

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[df_meta["category"] == "orig", "variable"].values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready", "derive"])].reset_index()


# --- Feature engineering -----------------------------------------------------------------------------------------

# Todo

# Check
print(setdiff(df_meta["variable"].values, df.columns.values))


# --- Define target and train/test-fold ----------------------------------------------------------------------------

# Target
df["target"] = df["income"].map({" - 50000.": 0, " 50000+.": 1})
df["target"].describe()

# Train/Test fold: usually split by time
np.random.seed(123)
df["fold"] = np.where(df["year"] == 95, "test", "train")
df["fold"][df.query("fold == 'train'").sample(frac = 0.1).index.values] = "util"
print(df["fold"].value_counts())
df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data
df["encode_flag"] = df["fold"].replace({"train": 0, "test": 0, "util": 1})  # Used for encoding

# Define the id
df = df.reset_index()
df["id"] = df.index.values + 1


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------

metr = df_meta_sub.loc[df_meta_sub["type"] == "metr", "variable"].values
df = Convert(features = metr, convert_to = "float").fit_transform(df)
df[metr].describe()

# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
df[metr + "_BINNED"] = df[metr]
df = Binning(features = metr + "_BINNED").fit_transform(df)
# Alternative: df[metr + "_BINNED"] = Binning(features=metr).fit_transform(df[metr].copy())

# Convert missings to own level ("(Missing)")
df[metr + "_BINNED"] = df[metr + "_BINNED"].fillna("(Missing)")
print(create_values_df(df[metr + "_BINNED"], 11))

# Get binned variables with just 1 bin (removed later)
onebin = (metr + "_BINNED")[df[metr + "_BINNED"].nunique() == 1]

# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------

# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending = False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
metr = setdiff(metr, remove)  # adapt metadata

# Check for outliers and skewness
df[metr].describe()
plot_distr(df, metr, target_type = TARGET_TYPE,
           color = color, ylim = ylim,
           ncol = 4, nrow = 2, w = 18, h = 12, pdf = plotloc + "census_distr_metr.pdf")

# Winsorize (hint: plot again before deciding for log-trafo)
df = Winsorize(features = metr, lower_quantile = 0.001, upper_quantile = 0.999).fit_transform(df)

# Log-Transform
tolog = np.array(["divdends_from_stocks"], dtype = "object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
metr = np.where(np.isin(metr, tolog), metr + "_LOG_", metr)  # adapt metadata (keep order)
df.rename(columns = dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace = True)  # adapt binned version


# --- Final variable information ------------------------------------------------------------------------------------

# Univariate variable importance
varimp_metr = calc_imp(df, np.append(metr, metr + "_BINNED"), target_type = TARGET_TYPE)
print(varimp_metr)

# Plot
plot_distr(df, features = np.column_stack((metr, metr + "_BINNED")).ravel(), target_type = TARGET_TYPE,
           varimp = varimp_metr, color = color, ylim = ylim,
           ncol = 4, nrow = 2, w = 24, h = 18, pdf = plotloc + "census_distr_metr_final.pdf")

# --- Removing variables -------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df, metr, cutoff = cutoff_corr, pdf = plotloc + "census_corr_metr.pdf")
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)

# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_fold = calc_imp(df, metr, "fold_num")

# Plot: only variables with with highest importance
metr_toprint = varimp_metr_fold[varimp_metr_fold > cutoff_varimp].index.values
plot_distr(df, metr_toprint, target = "fold_num", target_type = "CLASS",
           varimp = varimp_metr_fold,
           ncol = 2, nrow = 2, w = 12, h = 8, pdf = plotloc + "census_distr_metr_folddep.pdf")

# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------

miss = metr[df[metr].isnull().any().values]  # alternative: [x for x in metr if df[x].isnull().any()]


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------

# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["nomi", "ordi"]), "variable"].values
df = Convert(features = cate, convert_to = "str").fit_transform(df)
df[cate].describe()

# Convert ordinal features to make it "alphanumerically sorted"
ordi = ['industry_code', 'occupation_code', 'own_business_or_self_employed', 'veterans_benefits']
df[ordi] = df[ordi].apply(lambda x: x.str.zfill(2))


# --- Handling factor values ----------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 10
levinfo = df[cate].nunique().sort_values(ascending = False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features = cate, encode_flag_column = "encode_flag", target = "target").fit_transform(df)

# Convert toomany features: lump levels and map missings to own level
df = MapToomany(features = toomany, n_top = 10).fit_transform(df)

# Univariate variable importance
varimp_cate = calc_imp(df, np.append(cate, ["MISS_" + miss]), target_type = TARGET_TYPE)
print(varimp_cate)

# Check
plot_distr(df, np.append(cate, ["MISS_" + miss]), target_type = TARGET_TYPE,
           varimp = varimp_cate, color = color, ylim = ylim, min_width = min_width,
           nrow = 2, ncol = 3, w = 18, h = 12,
           pdf = plotloc + "census_distr_cate.pdf")  # maybe plot miss separately

# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove leakage variables
cate = setdiff(cate, ["xxx", "xxx"])
toomany = setdiff(toomany, ["xxx", "xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df.sample(n=int(1e4)), np.append(cate, ["MISS_" + miss]), cutoff = cutoff_corr, n_cluster = 5,
          w = 18, h= 18, pdf = plotloc + "census_corr_cate.pdf")


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_fold = calc_imp(df, np.append(cate, ["MISS_" + miss]), "fold_num")

# Plot: only variables with with highest importance
cate_toprint = varimp_cate_fold[varimp_cate_fold > cutoff_varimp].index.values
plot_distr(df, cate_toprint, target = "fold_num", target_type = "CLASS",
           varimp = varimp_cate_fold,
           ncol = 2, nrow = 2, w = 12, h = 8, pdf = plotloc + "census_distr_cate_folddep.pdf")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Adapt target ----------------------------------------------------------------------------------------

target_labels = "target"


# --- Define final features ----------------------------------------------------------------------------------------

# Standard: for xgboost or Lasso
metr_standard = np.append(metr, toomany + "_ENCODED")
cate_standard = np.append(cate, "MISS_" + miss)

# Binned: for Lasso
metr_binned = np.array([])
cate_binned = np.append(setdiff(metr + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
metr_encoded = np.concatenate([metr, cate + "_ENCODED", "MISS_" + miss + "_ENCODED"])
cate_encoded = np.array([])

# Check
all_features = np.unique(np.concatenate([metr_standard, cate_standard, metr_binned, cate_binned, metr_encoded]))
setdiff(all_features, df.columns.values.tolist())
setdiff(df.columns.values.tolist(), all_features)


# --- Remove burned data ----------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop = True)


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig = "all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open("census_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "target_labels": target_labels,
                 "metr_standard": metr_standard,
                 "cate_standard": cate_standard,
                 "metr_binned": metr_binned,
                 "cate_binned": cate_binned,
                 "metr_encoded": metr_encoded,
                 "cate_encoded": cate_encoded},
                file)
