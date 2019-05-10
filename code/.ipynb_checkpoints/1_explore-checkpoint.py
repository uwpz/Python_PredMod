# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

#######################################################################################################################-
# Libraries + Parallel Processing Start ----
#######################################################################################################################-

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate, RepeatedKFold, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

#import os
#os.chdir("C:/Users/Uwe/Desktop/python")
exec(open("./code/0_init.py").read())

sns.set(style="whitegrid")


#######################################################################################################################-
# Parameters ----
#######################################################################################################################-

dataloc = "./data/"
plotloc = "./output/"


#######################################################################################################################-
# My Functions ----
#######################################################################################################################-

def setdiff(a, b):
    return [x for x in a if x not in set(b)]


#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

#df.orig = read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE)
df_orig = pd.read_csv(dataloc + "titanic.csv")

#skip = function() {
'''
  # Check some stuff
  #summary(df.orig)
  df_orig.describe()
  df_orig.describe(include = ["object"])
  #df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  
  pd.concat([df_orig[catname].value_counts()[:4].reset_index().rename(columns = {"index": catname, catname: ""})
                     for catname in df_orig.select_dtypes(["object"]).columns.values], axis = 1)

  #table(df.tmp$survived) / nrow(df.tmp)
  df_orig["survived"].value_counts() / df_orig.shape[0]
#}
'''


# "Save" original data
#df = df.orig 
df = df_orig.copy()



# Feature engineering -------------------------------------------------------------------------------------------------------------

#df$deck = as.factor(str_sub(df$cabin, 1, 1))
df["deck"] = df["cabin"].str[:1]
#summary(df$deck)
df.deck.describe()
df.deck.value_counts()
# also possible: df$familysize = df$sibsp + df$parch as well as something with title from name




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
#df = mutate(df, target = factor(ifelse(survived == 0, "N", "Y"), levels = c("N","Y")),
#            target_num = ifelse(target == "N", 0 ,1))
df["target"] = np.where(df.survived == 0, "N", "Y") 
df["target_num"] = df.target.map({"N":0, "Y":1})
#summary(df[c("target","target_num")])
df[["target","target_num"]].describe(include = "all")


# Train/Test fold: usually split by time
#df$fold = factor("train", levels = c("train", "test"))
df["fold"] = "train"
#set.seed(123)
#df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test"
df.loc[df.sample(frac = 0.3, random_state = 123).index, "fold"] = "test"

#summary(df$fold)
df.fold.value_counts()




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------

#metr = c("age","fare")
metr = ["age","fare"]
#summary(df[metr]) 
df[metr].describe()




# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

#metr_binned = paste0(metr,"_BINNED_")
metr_binned = [x + "_BINNED_" for x in metr]
#df[metr_binned] = map(df[metr], ~ {
#  cut(., unique(quantile(., seq(0,1,0.1), na.rm = TRUE)), include.lowest = TRUE)
#})
df[metr_binned] = df[metr].apply(lambda x: pd.qcut(x, 10).astype(object))
df[metr_binned].describe()

# Convert missings to own level ("(Missing)")
#df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
df[metr_binned] = df[metr_binned].fillna("(missing)")
#summary(df[metr_binned],11)
df[metr_binned].describe()
{print(df[x].value_counts()[:11]) for x in metr_binned}



# Handling missings ----------------------------------------------------------------------------------------------

# Remove covariates with too many missings from metr 
#misspct = map_dbl(df[metr], ~ round(sum(is.na(.)/nrow(df)), 3)) #misssing percentage
misspct = df[metr].isnull().mean().round(3)
#misspct[order(misspct, decreasing = TRUE)] #view in descending order
misspct.sort_values(ascending = False)
remove = misspct[misspct > 0.95].index.values.tolist(); print(remove)
metr = setdiff(metr, remove)


# Create mising indicators
#(miss = metr[map_lgl(df[metr], ~ any(is.na(.)))])
miss = np.array(metr)[df[metr].isnull().any().values].tolist()
#df[paste0("MISS_",miss)] = map(df[miss], ~ as.factor(ifelse(is.na(.x), "miss", "no_miss")))
df[["MISS_" + x for x in miss]] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
                                  #df[miss].apply(lambda x: np.where(x.isnull(), "miss", "no_miss"))
#summary(df[,paste0("MISS_",miss)])
df[["MISS_" + x for x in miss]].describe()

# Impute missings with randomly sampled value (or median, see below)
#df[miss] = map(df[miss], ~ {
#  i.na = which(is.na(.x))
#  .x[i.na] = sample(.x[-i.na], length(i.na) , replace = TRUE)
#  #.x[i.na] = median(.x[-i.na], na.rm = TRUE) #median imputation
#  .x }
#)
df[miss] = df[miss].fillna(df[miss].median())
df[miss].describe()



# Outliers + Skewness --------------------------------------------------------------------------------------------

# Check for outliers and skewness
#plots = get_plot_distr_metr_class(df, metr, missinfo = NULL)
#ggsave(paste0(plotloc, "titanic_distr_metr.pdf"), suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2)), 
#       width = 18, height = 12)

#plt.figure()
fig, ax = plt.subplots(1,2)
for i in range(len(metr)):
    sns.distplot(df.loc[df.target == "Y", metr[i]], color = "red", label = "Y", ax = ax[i])
    sns.distplot(df.loc[df.target == "N", metr[i]], color = "blue", label = "N", ax = ax[i])
    ax[i].set_title(metr[i])
    ax[i].set_ylabel("density")
    ax[i].set_xlabel(metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
ax[0].legend(title = "Target", loc = "best")
fig.savefig(plotloc + "metr.pdf")
plt.close(fig)
#plt.figure()
metr = ["age","fare"]*10
fig, ax = plt.subplots(5,4)
for i in range(len(metr)):
    axact = ax.flat[i]
    sns.distplot(df.loc[df.target == "Y", metr[i]], color = "red", label = "Y", ax = axact)
    sns.distplot(df.loc[df.target == "N", metr[i]], color = "blue", label = "N", ax = axact)
    axact.set_title(metr[i])
    axact.set_ylabel("density")
    axact.set_xlabel(metr[i] + "(NA: " + str(round(misspct[metr[i]] * 100, 1)) + "%)")
ax[0,0].legend(title = "Target", loc = "best")
plt.subplot_tool()

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
plt.close()

# Winsorize
#df[,metr] = map(df[metr], ~ {
#  q_lower = quantile(., 0.01, na.rm = TRUE)
#  q_upper = quantile(., 0.99, na.rm = TRUE)
#  .[. < q_lower] = q_lower
#  .[. > q_upper] = q_upper
#  . }
#)
df[metr] = df[metr].apply(lambda x: winsorize(x, (0.05)))
df[metr].describe()

# Log-Transform
#tolog = c("fare")
tolog = ["fare"]
#df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
df[[x + "_LOG_" for x in tolog]] = df[tolog].apply(lambda x: np.where(x == 0, np.log(x+1), np.log(x)))
#metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metr and keep order
metr = [x + "_LOG_" if x in tolog else x for x in metr]
# alternative: metr = list(map(lambda x: x + "_LOG_" if x in tolog else x, metr))
#names(misspct) = metr #adapt misspct names
misspct.index = metr



# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance
#varimp = filterVarImp(df[metr], df$target, nonpara = TRUE) %>% 
#  mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .$Y
#names(varimp) = metr
#varimp[order(varimp, decreasing = TRUE)]

# Plot 
#plots = get_plot_distr_metr_class(df, metr, missinfo = misspct, varimpinfo = varimp)
#ggsave(paste0(plotloc, "titanic_distr_metr_final.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), width = 18, height = 12)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self predictors
#metr = setdiff(metr, "T3")
metr = setdiff(metr, ["xxx"])

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
#summary(df[metr])
#plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0)
m_corr = abs(df[metr].corr(method = "spearman"))
fig = sns.heatmap(m_corr, annot=True, fmt=".2f", cmap = "Blues").get_figure()
#ggsave(paste0(plotloc, "titanic_corr_metr.pdf"), plot, width = 8, height = 8)
fig.savefig(plotloc + "titanic_corr_metr.pdf")
plt.close(fig)
#metr = setdiff(metr, c("xxx")) #Put at xxx the variables to remove
metr = setdiff(metr, ["xxx"])
#metr_binned = setdiff(metr_binned, c("xxx_BINNED_")) #Put at xxx the variables to remove
metr_binned = setdiff(metr_binned, ["xxx_BINNED_"])



#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

#nomi = c("pclass","sex","sibsp","parch","deck","embarked","boat","home.dest")
nomi = ["pclass","sex","sibsp","parch","deck","embarked","boat","home.dest"]
#nomi = union(nomi, paste0("MISS_",miss)) #Add missing indicators
nomi = nomi + ["MISS_" + x for x in miss]
#df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))
df[nomi] = df[nomi].astype(object)
#summary(df[nomi])
df[nomi].describe(include = "all")




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
#df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
df[nomi] = df[nomi].fillna("(Missing)")
df[nomi].describe(include = "all")

# Create compact covariates for "too many members" columns 
topn_toomany = 20
#levinfo = map_int(df[nomi], ~ length(levels(.))) 
levinfo = df[nomi].apply(lambda x: x.unique().size)
#levinfo[order(levinfo, decreasing = TRUE)]
levinfo.sort_values(ascending = False)
#(toomany = names(levinfo)[which(levinfo > topn_toomany)])
toomany = levinfo[levinfo > topn_toomany].index.values.tolist(); print(toomany)
#(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables
toomany = setdiff(toomany, ["xxx"]) #Set exception for important variables
#df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(fct_infreq(.), topn_toomany, other_level = "_OTHER_")) #collapse
#df[[x + "_OTHER_" for x in toomany]] = df[toomany].apply(lambda x: 
#    np.where(x.isin(x.value_counts()[0:topn_toomany].index.tolist()), x, "_OTHER_"))
df[[x + "_OTHER_" for x in toomany]] = df[toomany].apply(lambda x: 
    x.replace(x.value_counts()[topn_toomany:].index.values, "_OTHER_"))
#nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #Exchange name
nomi = [x + "_OTHER_" if x in toomany else x for x in nomi]

#summary(df[nomi], topn_toomany + 2)
df[nomi].apply(lambda x: print(x.value_counts()))

# Univariate variable importance
#varimp = filterVarImp(df[nomi], df$target, nonpara = TRUE) %>% 
  #mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .$Y
#names(varimp) = nomi
#varimp[order(varimp, decreasing = TRUE)]


# Check
#plots = get_plot_distr_nomi_class(df, nomi, varimpinfo = varimp)
#ggsave(paste0(plotloc, "titanic_distr_nomi.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)
fig, ax = plt.subplots(3,4, figsize=(18, 16))
for i in range(len(nomi)):
    df_tmp = pd.DataFrame({"h": df.groupby(nomi[i])["target_num"].mean(), 
                           "w": df.groupby(nomi[i]).size()}).reset_index()
    df_tmp["w"] = df_tmp["w"]/max(df_tmp["w"])
    axact = ax.flat[i]
    sns.barplot(df_tmp.h, df_tmp[nomi[i]], orient = "h", color = "coral", ax = axact)
    axact.set_xlabel("Proportion Target = Y")
    axact.axvline(np.mean(df.target_num), ls = "dotted", color = "black")
    for bar,width in zip(axact.patches, df_tmp.w):
        bar.set_height(width)
plt.subplots_adjust(wspace = 1)
fig.savefig(plotloc + "nomi.pdf", dpi = 600)

plt.close(fig)


# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-predictors
#nomi = setdiff(nomi, "boat_OTHER_")
nomi = setdiff(nomi, ["boat_OTHER_"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
#plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0)
#ggsave(paste0(plotloc, "titanic_corr_nomi.pdf"), plot, width = 12, height = 12)
#nomi = setdiff(nomi, "xxx")
nomi = setdiff(nomi, ["xxx"])

df[nomi] = df[nomi].apply(lambda x: x.astype("category"))


#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final predictors ----------------------------------------------------------------------------------------

#predictors = c(metr, nomi)
predictors = metr + nomi
#formula = as.formula(paste("target", "~", paste(predictors, collapse = " + ")))
#predictors_binned = c(metr_binned, setdiff(nomi, paste0("MISS_",miss))) #do not need indicators if binned variables
predictors_binned = metr_binned + setdiff(nomi, ["MISS_" + x for x in miss])
#formula_binned = as.formula(paste("target", "~", paste(predictors_binned, collapse = " + ")))

# Check
#summary(df[predictors])
df[predictors].describe(include = "all")
#setdiff(predictors, colnames(df))
setdiff(predictors, df.columns.values)
df[predictors_binned].describe(include = "all")
#setdiff(predictors_binned, colnames(df))
setdiff(predictors_binned, df.columns.values)




# Save image ----------------------------------------------------------------------------------------------------------
#rm(df.orig)
del df_orig
del p
#rm(plots)
#save.image("1_explore.rdata")
dill.dump_session("1_explore.pkl") 



