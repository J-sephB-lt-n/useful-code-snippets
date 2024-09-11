"""
TAGS: fit|learning|machine|machine learning|ml|model|models|predict|predictive|regression|scikit-learn|sklearn|train
DESCRIPTION: An example of a regression pipeline in scikit-learn
REQUIREMENTS: pip install 'pandas==2.2.2' 'polars==1.6.0' 'pyarrow==17.0.0' 'scikit-learn==1.5.1' 'seaborn==0.13.2' 'shap==0.46.0'
NOTES: In a future iteration, I want to include a hyperparameter tuning step in this script
NOTES: The pipeline currently does the following:
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # I will drop pandas when sklearn integrates polars more fully
import polars as pl
import seaborn as sns
import shap
from sklearn import linear_model
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

pl.Config.set_tbl_cols(15)  # show up to 15 columns in table display

df = (
    # COLUMNS:
    #       price:      price in US dollars (\$326--\$18,823)
    #       carat:      weight of the diamond (0.2--5.01)
    #       cut:        quality of the cut (Fair, Good, Very Good, Premium, Ideal)
    #       color:      diamond colour, from J (worst) to D (best)
    #       clarity:    a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
    #       x:          length in mm (0--10.74)
    #       y:          width in mm (0--58.9)
    #       z:          depth in mm (0--31.8)
    #       depth:      total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
    #       table:      width of top of diamond relative to widest point (43--95)
    pl.scan_csv("diamonds_price_predict.csv")
    .drop("")  # drop this unnamed index column
    .rename(
        {
            "x": "length",
            "y": "width",
            "z": "depth",
            "depth": "depth_percent",
        }
    )
)

# quick peek at the data #
df.head(10).collect()
df.describe()

# data splitting #
X = df.drop("price")
y = df.select("price")
X = X.collect().to_pandas()  # I'm not happy about this
y = y.collect().to_pandas()["price"].to_numpy()  # I'm not happy about this
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=69420
)

# data pre-processing #
numeric_transformer = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)
numeric_splines_transformer = Pipeline(
    steps=[
        # ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("scaler", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                n_knots=5,
                degree=3,
                knots="quantile",
                extrapolation="constant",
                include_bias=True,
            ),
        ),
    ]
)
data_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            make_column_selector(dtype_include=np.number),
        ),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=object),
        ),
    ]
)
splines_data_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_splines_transformer,
            make_column_selector(dtype_include=np.number),
        ),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=object),
        ),
    ]
)

# define some models to compare #
pipelines = {
    "ridge_splines": Pipeline(
        steps=[
            ("preprocess_data", splines_data_preprocessor),
            ("feature_selection", VarianceThreshold(threshold=0.0)),
            ("regressor", linear_model.Ridge()),
        ]
    ),
    "gbm_hist": Pipeline(
        steps=[
            ("preprocess_data", data_preprocessor),
            ("feature_selection", VarianceThreshold(threshold=0.0)),
            ("regressor", HistGradientBoostingRegressor()),
        ]
    ),
    "extremely_randomized_trees": Pipeline(
        steps=[
            ("preprocess_data", data_preprocessor),
            ("feature_selection", VarianceThreshold(threshold=0.0)),
            ("regressor", ExtraTreesRegressor()),
        ]
    ),
}

# cross validation #
cross_valid_results = {}
for pipeline_name, pipeline in pipelines.items():
    print(f"Started: [{pipeline_name}]")
    cross_valid_results[pipeline_name] = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=[
            "r2",  # explain
            "max_error",  # max( |y_true-y_pred| )
            "neg_mean_absolute_error",  # mean( |y_true-y_pred| )
            "neg_root_mean_squared_error",  # explain
            "neg_mean_absolute_percentage_error",  # mean( |y_true-y_pred| / |y_true| )
        ],
    )
    print(
        f"Finished [{pipeline_name}] in "
        f'{(cross_valid_results[pipeline_name]["fit_time"].sum() + cross_valid_results[pipeline_name]["score_time"].sum()):,.0f}'
        " seconds"
    )

# visualise cross validation results #
cross_valid_results_long = []
for model, metrics in cross_valid_results.items():
    for metric, values in metrics.items():
        for value in values:
            cross_valid_results_long.append(
                {"model": model, "metric": metric, "value": value}
            )
g = sns.FacetGrid(
    pd.DataFrame(cross_valid_results_long),
    col="metric",
    hue="model",
    sharex=False,
    sharey=False,
    height=4,
    aspect=0.8,
    col_wrap=4,
)
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
g.map(sns.stripplot, "model", "value", jitter=True, dodge=True)
g.set_titles("{col_name}")
g.set_axis_labels("Model", "Metric Value")
g.tight_layout()
plt.show()


# select final model and train/predict #
final_model: Pipeline = pipelines["extremely_randomized_trees"]
fit_start_time: float = time.perf_counter()
final_model.fit(X_train, y_train)
fit_end_time: float = time.perf_counter()
print(f"Model finished training in {(fit_end_time-fit_start_time):,.2f} seconds")
preds_traindata: np.ndarray = final_model.predict(X_train)
preds_testdata: np.ndarray = final_model.predict(X_test)

# visualise distributions of prediction errors #
errors_traindata: np.ndarray = preds_traindata - y_train
errors_testdata: np.ndarray = preds_testdata - y_test
percentage_errors_traindata: np.ndarray = (preds_traindata - y_train) / y_train
percentage_errors_testdata: np.ndarray = (preds_testdata - y_test) / y_test
sns.histplot(errors_traindata, bins=100, kde=True, color="red")
plt.title(r"Distribution of prediction errors ($\hat{y} - y$) on Training data")
plt.xlabel("Error ($\hat{y}-y$)")
plt.show()
sns.histplot(errors_testdata, bins=100, kde=True, color="red")
plt.title(r"Distribution of prediction errors ($\hat{y} - y$) on Test (unseen) data")
plt.xlabel("Error ($\hat{y}-y$)")
plt.show()
sns.histplot(percentage_errors_traindata, bins=100, kde=True, color="red")
plt.title(
    r"Distribution of prediction % errors ($\frac{\hat{y} - y}{y}$) on Training data"
)
plt.xlabel(r"% Error ($\frac{\hat{y}-y}{y}$)")
plt.show()
sns.histplot(percentage_errors_testdata, bins=100, kde=True, color="red")
plt.title(
    r"Distribution of prediction % errors ($\frac{\hat{y} - y}{y}$) on Test (unseen) data"
)
plt.xlabel(r"% Error ($\frac{\hat{y}-y}{y}$)")
plt.show()
