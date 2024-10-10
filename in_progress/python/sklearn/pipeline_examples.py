"""
TAGS: 
DESCRIPTION: 
REQUIREMENTS: pip install scikit-learn 
"""

from sklearn import ensemble, linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

housing_data = fetch_california_housing()
X_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = housing_data.target

scaler_splines_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                n_knots=5,
                degree=3,
                knots="quantile",
                extrapolation="linear",
                include_bias=True,
            ),
        ),
    ]
)
one_hot_scaler_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
        ("scaler", StandardScaler()),
    ]
)
binning_scaler_pipeline = Pipeline(
    steps=[
        (
            "binning",
            KBinsDiscretizer(
                n_bins=10,
                encode="onehot-dense",
                strategy="quantile",
            ),
        ),
        ("scaler", StandardScaler()),
    ]
)
