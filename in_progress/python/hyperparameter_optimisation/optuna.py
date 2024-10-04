"""
TAGS: optim|optimisation|optimization|hyperparameter|hyperparameter optimisation|optuna
DESCRIPTION: TODO
REQUIREMENTS: pip install optuna scikit-learn seaborn 
"""

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing_data = fetch_california_housing()
X_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = housing_data.target

# turn latitude and longitude into blocks
# (because I want to use a linear model, which won't handle these features well)
X_df["Latitude_block"] = pd.qcut(X_df["Latitude"], q=10, labels=None)
X_df["Longitude_block"] = pd.qcut(X_df["Longitude"], q=10, labels=None)
# X_df["Latitude_block"] = pd.cut(X_df["Latitude"], bins=10, labels=None)
# X_df["Longitude_block"] = pd.cut(X_df["Longitude"], bins=10, labels=None)
X_df["geo_block"] = (
    X_df["Latitude_block"].astype(str) + "_" + X_df["Longitude_block"].astype(str)
)
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="Longitude",
    y="Latitude",
    hue="geo_block",
    style="geo_block",
    data=X_df,
    legend=False,
)
lat_deciles = np.quantile(X_df["Latitude"], np.arange(0.1, 1.0, 0.1))
lon_deciles = np.quantile(X_df["Longitude"], np.arange(0.1, 1.0, 0.1))
for lat in lat_deciles:
    plt.axhline(lat, color="black", alpha=0.7)
for lon in lon_deciles:
    plt.axvline(lon, color="black", alpha=0.7)
plt.show()
X_df = X_df.drop(["Latitude", "Longitude", "Latitude_block", "Longitude_block"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.1,
    random_state=80085,
)
