"""
TAGS: optim|optimisation|optimization|hyperparameter|hyperparameter optimisation|optuna
DESCRIPTION: TODO
REQUIREMENTS: pip install optuna scikit-learn seaborn 
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn.base
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

housing_data = fetch_california_housing()
X_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = housing_data.target

# turn latitude and longitude into blocks
# (to use for linear models, which won't handle these features well)
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
# X_df = X_df.drop(["Latitude", "Longitude", "Latitude_block", "Longitude_block"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.1,
    random_state=80085,
)


def build_pipeline(
    include_standard_scaler: bool,
    include_splines: bool,
    spline_degree: Optional[int],
    spline_n_knots: Optional[int],
    include_interaction_terms: bool,
    model: sklearn.base.RegressorMixin,
) -> Pipeline:
    """Returns a regression pipeline with the specified components and regression model"""
    transformers = []  # for data preprocessing

    if isinstance(model, linear_model.Ridge):
        numeric_colnames: list[str] = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
        ]
        categorical_colnames: list[str] = [
            "geo_block",
        ]
    elif isinstance(model, HistGradientBoostingRegressor):
        numeric_colnames: list[str] = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        categorical_colnames: list[str] = []
    else:
        raise ValueError("model type could not be identified")

    if numeric_colnames:
        numeric_transformers = []
        if include_standard_scaler:
            numeric_transformers.append(("scaler", StandardScaler()))
        if include_splines:
            numeric_transformers.append(
                (
                    "splines",
                    SplineTransformer(
                        degree=spline_degree,
                        n_knots=spline_n_knots,
                        knots="quantile",
                        extrapolation="linear",
                        include_bias=True,
                    ),
                )
            )
        if include_interaction_terms:
            numeric_transformers.append(
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2,
                        interaction_only=True,
                        include_bias=False,
                    ),
                )
            )
        transformers.append(
            (
                "numeric",
                Pipeline(numeric_transformers),
                numeric_colnames,
            )
        )

    if categorical_colnames:
        categorical_transformers = []
        categorical_transformers.append(
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            )
        )
        transformers.append(
            (
                "categorical",
                Pipeline(categorical_transformers),
                categorical_colnames,
            )
        )

    return Pipeline(
        steps=[
            (
                "preprocess_data",
                ColumnTransformer(transformers),
            ),
            (
                "model",
                model,
            ),
        ]
    )


model_pipeline: Pipeline = build_pipeline(
    include_standard_scaler=True,
    include_splines=True,
    spline_degree=3,
    spline_n_knots=10,
    include_interaction_terms=True,
    model=HistGradientBoostingRegressor(),
)
