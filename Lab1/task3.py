# %% [markdown]
# Hemnet data

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the Excel file using Pandas.
alldata = pd.read_excel("Hemnet_data.xlsx")

# # Convert the timestamp string to an integer representing the year.
alldata["year"] = pd.DatetimeIndex(alldata["Sold Date"]).year

# Convert 'yes' to 1 and 'no' to 0
alldata["Balcony"] = alldata["Balcony"].str.lower().map({"yes": 1, "no": 0})
alldata["Patio"] = alldata["Patio"].str.lower().map({"yes": 1, "no": 0})
alldata["Lift"] = alldata["Lift"].str.lower().map({"yes": 1, "no": 0})

# Select the 12 input columns and the output column.
selected_columns = [
    "Final Price (kr)",
    "year",
    "Num of Room",
    "Living Area (m²)",
    "Balcony",
    "Patio",
    "Current Floor",
    "Total Floor",
    "Lift",
    "Built Year",
    "Fee (kr/month)",
    "Operating Fee (kr/year)",
]
alldata = alldata[selected_columns]
alldata["Fee (kr/month)"] = (
    alldata["Fee (kr/month)"]
    .astype(str)
    .str.replace("kr", "", case=False)
    .str.replace(" ", "")
)
alldata["Fee (kr/month)"] = pd.to_numeric(alldata["Fee (kr/month)"], errors="coerce")
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop("Final Price (kr)", axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled["Final Price (kr)"].apply(np.log)

# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

classifiers = (
    LinearRegression,
    Ridge,
    Lasso,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    MLPRegressor,
)

# Test each classifier and print the mean cross-validation score.
for classifier in classifiers:
    clf = classifier()
    print(
        f"{clf} {cross_validate(clf, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()}"
    )


# %% [markdown]
# The RandomForestRegressor performs the best with nmse around -0.148.
# The GradientBoosting performs around -0.158.
# We therefore use RandomForestRegressor for further evaluation.


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

regr = RandomForestRegressor()
regr.fit(Xtrain, Ytrain)
mean_squared_error(Ytest, regr.predict(Xtest))

# %% [markdown]
# Our final mse is around 0.136
