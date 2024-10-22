"""
TAGS: classification|pipeline|regression|scikit-learn|sklearn|supervised|supervised learning
DESCRIPTION: Example usage of pipelines in scikit-learn
REQUIREMENTS: pip install scikit-learn 
"""

import numpy as np
import pandas as pd
from sklearn import dummy, ensemble, linear_model
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

housing_data = fetch_california_housing()
X_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = housing_data.target

scaler_splines_feature_pipeline = Pipeline(
    # scales the feature the creates splines #
    steps=[
        ("scaler", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                n_knots=10,
                degree=3,
                knots="quantile",
                extrapolation="linear",
                include_bias=True,
            ),
        ),
    ]
)
one_hot_feature_pipeline = Pipeline(
    # 1-hot encodes the feature #
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)
onehot_scaler_feature_pipeline = Pipeline(
    # 1-hot encodes the feature then scales it #
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
        ("scaler", StandardScaler()),
    ]
)
binning_onehot_scaler_feature_pipeline = Pipeline(
    # bins the feature, then 1-hot encodes it, then scales it #
    steps=[
        (
            "binning",
            KBinsDiscretizer(
                n_bins=10,
                encode="onehot-dense",
                strategy="quantile",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
        ("scaler", StandardScaler()),
    ]
)
do_nothing_feature_pipeline = FunctionTransformer(
    # passes the feature on as is #
    lambda x: x
)

hist_gbm_data_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric_cols",
            do_nothing_feature_pipeline,
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
        ),
        (
            "categorical_cols",
            one_hot_feature_pipeline,
            [],
        ),
    ]
)

ridge_data_preprocessor = ColumnTransformer(
    transformers=[
        (
            "lat_long",
            binning_onehot_scaler_feature_pipeline,
            [
                "Latitude",
                "Longitude",
            ],
        ),
        (
            "numeric_cols",
            scaler_splines_feature_pipeline,
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
            ],
        ),
    ]
)

X_df_train, X_df_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.1,
    random_state=8008135,
)

dummy_baseline_pipeline = Pipeline(
    steps=[
        (
            "regressor",
            dummy.DummyRegressor(),
        )
    ]
)
hist_gbm_pipeline = Pipeline(
    steps=[
        ("preprocess_data", hist_gbm_data_preprocessor),
        ("regressor", ensemble.HistGradientBoostingRegressor()),
    ]
)
ridge_pipeline = Pipeline(
    steps=[
        ("preprocess_data", ridge_data_preprocessor),
        ("regressor", linear_model.RidgeCV(alphas=np.logspace(-10, 10, 21))),
    ]
)

dummy_baseline_pipeline.fit(X_df_train, y_train)
hist_gbm_pipeline.fit(X_df_train, y_train)
ridge_pipeline.fit(X_df_train, y_train)

dummy_baseline_pipeline_test_preds = dummy_baseline_pipeline.predict(X_df_test)
hist_gbm_pipeline_test_preds = hist_gbm_pipeline.predict(X_df_test)
ridge_pipeline_test_preds = ridge_pipeline.predict(X_df_test)

test_set_mape = {
    "dummy_baseline": np.mean(
        np.abs(dummy_baseline_pipeline_test_preds - y_test) / y_test
    ),
    "hist_gbm": np.mean(np.abs(hist_gbm_pipeline_test_preds - y_test) / y_test),
    "ridge": np.mean(np.abs(ridge_pipeline_test_preds - y_test) / y_test),
}

print("--MEAN ABSOLUTE % ERROR (MAPE) ON TEST SET--")
for model_name, mape in test_set_mape.items():
    print(model_name, f"{mape:,.2f}")
