"""
TAGS: fit|learning|machine|machine learning|ml|model|models|predict|predictive|regression|scikit-learn|sklearn|train
DESCRIPTION: An example of a regression pipeline in scikit-learn
REQUIREMENTS: pip install 'pandas==2.2.2' 'polars==1.6.0' 'pyarrow==17.0.0' 'scikit-learn==1.5.1' 'seaborn==0.13.2' 'shap==0.46.0'
NOTES: In a future iteration, I want to include a hyperparameter tuning step in this script
"""

import itertools
import time
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # I will use only polars when there is more support for polars in sklearn
import polars as pl
import seaborn as sns
import shap
from sklearn import linear_model
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

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
print(
    "--First 10 rows--\n",
    df.head(10).collect(),
)
print(
    "--data summary--\n",
    df.describe(),
)

# data splitting #
X = df.drop("price")
y = df.select("price")
X = X.collect().to_pandas()  # I'm not happy about this
y = y.collect().to_pandas()["price"].to_numpy()  # I'm not happy about this
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=69420
)

numeric_feature_colnames: set[str] = set()
categorical_feature_colnames: set[str] = set()
for colname in X_test.columns:
    if X_test[colname].dtype.kind in "iufc":
        numeric_feature_colnames.add(colname)
    else:
        categorical_feature_colnames.add(colname)
print(
    f"""
Numeric features are: {", ".join(numeric_feature_colnames)}

Categorical features are: {", ".join(categorical_feature_colnames)}
"""
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
                extrapolation="linear",
                include_bias=True,
            ),
        ),
    ]
)
interaction_terms_transformer = Pipeline(
    steps=[
        (
            "interaction_terms",
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
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
            ("add_interaction_terms", interaction_terms_transformer),
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
k_folds: int = 10
for pipeline_name, pipeline in pipelines.items():
    print(f"Started: [{pipeline_name}]")
    cross_valid_results[pipeline_name] = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=k_folds,
        scoring=[
            "r2",  # R^2 = 'coefficient of determination' = 1 - sum(y_i-y^_i)^2 / sum(y_i-mean(y))^2
            "max_error",  # max( |y_true-y_pred| )
            "neg_mean_absolute_error",  # - mean( |y_true-y_pred| )
            "neg_root_mean_squared_error",  # - sqrt( mean( (y_true-y_pred)^2 ) )
            "neg_mean_absolute_percentage_error",  # - mean( |y_true-y_pred| / |y_true| )
        ],
    )
    print(
        f"\tFinished [{pipeline_name}] in "
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
g.figure.suptitle(f"Results of {k_folds}-Fold Cross Validation", fontsize=16)
g.tight_layout()
g.figure.subplots_adjust(top=0.9)  # Adjust top to make space for the title
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
absolute_errors_traindata = np.abs(errors_traindata)
absolute_errors_testdata = np.abs(errors_testdata)

sns.histplot(errors_traindata, bins=100, kde=True, color="red")
plt.title(r"Distribution of prediction errors ($\hat{y} - y$) on Training data")
plt.xlabel(r"Error ($\hat{y}-y$)")
plt.show()

sns.histplot(errors_testdata, bins=100, kde=True, color="red")
plt.title(r"Distribution of prediction errors ($\hat{y} - y$) on Test (unseen) data")
plt.xlabel(r"Error ($\hat{y}-y$)")
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

# look for univariate areas of the feature space with large errors #
for colname in numeric_feature_colnames:
    plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color="black", alpha=0.5)
    plt.scatter(X_test[colname], errors_testdata, alpha=0.2, s=5)
    plt.xlabel(colname)
    plt.ylabel("Prediction error")
    plt.title(
        f"Scatterplot of feature '{colname}' vs prediction_error (unseen test data)"
    )
    plt.show()

SHOW_OUTLIERS_ON_BOXPLOT: Final[bool] = False
for colname in categorical_feature_colnames:
    sns.boxplot(
        x=colname,
        y="error",
        data=pd.concat(
            [
                X_test.reset_index(),
                pd.DataFrame({"error": errors_testdata}),
            ],
            axis=1,
        ),
        showfliers=SHOW_OUTLIERS_ON_BOXPLOT,
    )
    plt.xlabel(colname)
    plt.ylabel("Prediction error")
    plt.title(
        f"Distribution of prediction errors within each level of feature '{colname}' (unseen test data) [displayOutliers={SHOW_OUTLIERS_ON_BOXPLOT}]"
    )
    plt.show()

# look for bivariate areas of the feature space with large errors #
case_both_numeric_features_colnames: list[tuple[str, str]] = []
case_both_categorical_colnames: list[tuple[str, str]] = []
case_one_numeric_one_categorical_colnames: list[tuple[str, str]] = []

for x1_name, x2_name in itertools.combinations(X_test.columns, 2):
    n_numeric_features: int = sum(
        [x in numeric_feature_colnames for x in (x1_name, x2_name)]
    )
    match n_numeric_features:
        case 2:
            case_both_numeric_features_colnames.append((x1_name, x2_name))
        case 1:
            case_one_numeric_one_categorical_colnames.append((x1_name, x2_name))
        case _:
            case_both_categorical_colnames.append((x1_name, x2_name))

X_test_high_absolute_errors_last = pd.concat(
    # I use this dataframe to plot higher errors on top of lower errors #
    [
        X_test.reset_index(drop=True),
        pd.DataFrame(
            {
                "error": errors_testdata,
                "absolute_error": np.abs(errors_testdata),
            }
        ),
    ],
    axis=1,
).sort_values("absolute_error")

for x1_name, x2_name in case_both_numeric_features_colnames:
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        X_test_high_absolute_errors_last[x1_name],
        X_test_high_absolute_errors_last[x2_name],
        c=X_test_high_absolute_errors_last["absolute_error"],
        cmap="viridis",
        # alpha=0.5,
        s=10,
    )
    plt.colorbar(scatter, label="absolute_error")
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)
    plt.title(
        f"Scatterplot of feature '{x1_name}' vs feature '{x2_name}' (unseen test data)"
    )
    plt.show()

for x1_name, x2_name in case_one_numeric_one_categorical_colnames:
    if x1_name in numeric_feature_colnames:
        numeric_feature_name: str = x1_name
        categ_feature_name: str = x2_name
    else:
        numeric_feature_name: str = x2_name
        categ_feature_name: str = x1_name
    g = sns.FacetGrid(
        X_test_high_absolute_errors_last,
        col=categ_feature_name,
        col_wrap=5,
        height=4,
        aspect=1,
    )
    g.map(
        sns.scatterplot,
        numeric_feature_name,
        "error",
        s=10,
        alpha=0.3,
    )
    g.figure.suptitle(
        (
            f"Distribution of prediction errors within feature '{numeric_feature_name}',"
            f" separately within each level of feature '{categ_feature_name}'"
        ),
    )
    g.figure.subplots_adjust(top=0.9)  # Adjust top to make space for the title
    g.set_axis_labels(numeric_feature_name, "Prediction error")
    plt.show()

HEATMAP_MIN_OBSERVATIONS_THRESHOLD: Final[int] = (
    # subsets of the data with fewer than this number of samples are omitted from the heatmap
    50
)
for x1_name, x2_name in case_both_categorical_colnames:
    heatmap_data = X_test_high_absolute_errors_last.pivot_table(
        index=x1_name, columns=x2_name, values="absolute_error", aggfunc="mean"
    )
    observation_counts = X_test_high_absolute_errors_last.pivot_table(
        index=x1_name, columns=x2_name, values="absolute_error", aggfunc="count"
    )
    mask = observation_counts >= HEATMAP_MIN_OBSERVATIONS_THRESHOLD
    filtered_heatmap_data = heatmap_data.where(mask)
    filtered_heatmap_data = filtered_heatmap_data.fillna(np.nan)
    sns.heatmap(
        filtered_heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Mean Absolute Error"},
    )
    plt.title(
        f"Heatmap of feature '{x1_name}' vs feature '{x2_name}' coloured by Mean Absolute Error"
        f"\n(cells with fewer than {HEATMAP_MIN_OBSERVATIONS_THRESHOLD} observations not populated)"
    )
    plt.xlabel(x2_name)
    plt.ylabel(x1_name)
    plt.show()


# investigate feature effects on prediction (SHAP values) #
def predict_for_kernel_shap(x_in: np.ndarray) -> np.ndarray:
    """A prediction function which Kernel SHAP can use
    (since it gets stuck on column names using the standard predict function)
    Notes:
        See https://datascience.stackexchange.com/questions/52476/how-to-use-shap-kernal-explainer-with-pipeline-models
    """
    x_df = pd.DataFrame(x_in, columns=X_train.columns)
    return final_model.predict(x_df)


X_train_sample = X_train.sample(100)
shap_explainer = shap.KernelExplainer(
    lambda x: final_model.predict(pd.DataFrame(x, columns=X_train.columns)),
    X_train_sample,
)
shap_values_start_time: float = time.perf_counter()
shap_values = shap_explainer(X_train_sample)
shap_values_end_time: float = time.perf_counter()
print(
    f"Finished calculating {len(X_train_sample)} SHAP values in {(shap_values_end_time-shap_values_start_time)/60:,.2f} minutes"
)
for feature_name in X_train:
    shap.dependence_plot(feature_name, shap_values.values, X_train_sample)
