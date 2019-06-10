# ######################################################################################################################
# Score
# ######################################################################################################################

# General libraries, parameters and functions
from init import *

# Load pipelines
with open("productive.pkl", "rb") as file:
    d_pipelines = pickle.load(file)

# Read scoring data
df = pd.read_csv(dataloc + "titanic.csv").iloc[1000:, :]

# Transform
df = d_pipelines["pipeline_etl"].transform(df)

# Fit
yhat = scale_predictions(d_pipelines["pipeline_fit"].predict_proba(df),
                         d_pipelines["pipeline_etl"].named_steps["undersample_n"].b_sample,
                         d_pipelines["pipeline_etl"].named_steps["undersample_n"].b_all)
