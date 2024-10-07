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
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
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
    """Returns a regression pipeline with the specified components and regression model

    Example:
        >>> model_pipeline: Pipeline = build_pipeline(
        ...     include_standard_scaler=True,
        ...     include_splines=True,
        ...     spline_degree=3,
        ...     spline_n_knots=10,
        ...     include_interaction_terms=True,
        ...     model=linear_model.Ridge(alpha=1.0),
        ... )
    """
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


def objective_func(trial):
    model_choice: str = trial.suggest_categorical("model", ["hist_gbm", "ridge"])
    if model_choice == "ridge":
        model = linear_model.Ridge(
            alpha=trial.suggest_float("alpha", 0, 100, log=True),
        )
        include_splines: bool = trial.suggest_categorical(
            "include_splines", [True, False]
        )
        spline_degree: Optional[int] = trial.suggest_int("spline_degree", 2, 5)
        spline_n_knots: Optional[int] = trial.suggest_int("spline_n_knots", 3, 20)
        include_interaction_terms: bool = trial.suggest_categorical(
            "interaction_terms", [True, False]
        )
    elif model_choice == "hist_gbm":
        limit_max_depth: bool = trial.suggest_categorical(
            "limit_max_depth", [True, False]
        )
        if limit_max_depth:
            max_depth = trial.suggest_int("max_depth", 1, 30)
        else:
            max_depth = None
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=trial.suggest_float("learning_rate", 0, 1),
            # n_estimators=
            max_depth=max_depth,
            # min_samples_split 0.1 to 1.0
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
        )
        include_splines: bool = False
        spline_degree: Optional[int] = None
        spline_n_knots: Optional[int] = None
        include_interaction_terms: bool = False
    else:
        raise ValueError(f"invalid model choice '{model_choice}'")
    model_pipeline: Pipeline = build_pipeline(
        include_standard_scaler=trial.suggest_categorical("scale_data", [True, False]),
        include_splines=include_splines,
        spline_degree=spline_degree,
        spline_n_knots=spline_n_knots,
        include_interaction_terms=include_interaction_terms,
        model=model,
    )
    cv_results = cross_validate(
        estimator=model_pipeline,
        X=X_train,
        y=y_train,
        cv=10,  # number of folds
        scoring=[
            # "r2",  # R^2 = 'coefficient of determination' = 1 - sum(y_i-y^_i)^2 / sum(y_i-mean(y))^2
            # "max_error",  # max( |y_true-y_pred| )
            # "neg_mean_absolute_error",  # - mean( |y_true-y_pred| )
            # "neg_root_mean_squared_error",  # - sqrt( mean( (y_true-y_pred)^2 ) )
            "neg_mean_absolute_percentage_error",  # - mean( |y_true-y_pred| / |y_true| )
        ],
        n_jobs=5,
    )
    return -float(cv_results["test_neg_mean_absolute_percentage_error"].mean())


optuna_study = optuna.create_study()
optuna_study.optimize(objective_func, n_trials=100)
