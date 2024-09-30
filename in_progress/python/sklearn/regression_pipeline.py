"""
TAGS: fit|learning|machine|machine learning|ml|model|models|predict|predictive|regression|scikit-learn|sklearn|train
DESCRIPTION: An example of a regression pipeline in scikit-learn
REQUIREMENTS: pip install pandas polars pyarrow scikit-learn seaborn shap # bottleneck numexpr
NOTES: In a future iteration, I want to include a hyperparameter tuning step in this script
"""

import itertools
import random
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
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

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


# some utility functions #
def univariate_prediction_plot(
    x: list[str | float | int],
    y_pred: list[str | float | int],
    y_true: list[str | float | int],
    x_colname: str,
    y_colname: str,
    metric_name: str,
    metric_value: int | float,
) -> None:
    """Takes in a 1-D list of values of a single feature `x` (either labels or numbers),
    a 1-D list of predictions `y_pred` (either labels or numbers) and a
    1-D list of true outcome values `y_true` (either labels or numbers) and
    displays a matplotlib visualisation of this data"""
    plt.figure(figsize=(10, 6))

    if isinstance(x[0], (int, float)) and isinstance(y_pred[0], (int, float)):
        plt.scatter(x, y_true, label="True", color="blue", alpha=1.0, s=5)
        plt.scatter(x, y_pred, label="Predicted", color="red", alpha=0.6, s=5)
        plt.xlabel(f"X ({x_colname})")
        plt.ylabel(f"Y ({y_colname})")
        plt.legend()
        plt.title(
            f"Performance of numeric X ({x_colname}) predicting numeric Y ({y_colname}): True vs Predicted\n"
            f"{metric_name}: {metric_value:,.2f}",
        )
    elif isinstance(x[0], (int, float)) and isinstance(y_pred[0], str):
        category_nums_map: dict[str, int] = {
            label: num for num, label in enumerate(set(y_true))
        }
        y_true_nums_jittered: list[float] = [
            category_nums_map[y_i] + random.uniform(-0.4, 0.4) for y_i in y_true
        ]
        mismatches_x = []
        mismatches_y = []
        for x_i, y_pred_i, y_true_i, y_true_num_jit_i in zip(
            x, y_pred, y_true, y_true_nums_jittered
        ):
            if y_pred_i != y_true_i:
                mismatches_x.append(x_i)
                mismatches_y.append(y_true_num_jit_i)
        plt.scatter(
            x,
            y_true_nums_jittered,
            color="blue",
            alpha=1.0,
            s=2,
            label="Correct Prediction",
        )
        plt.scatter(
            mismatches_x,
            mismatches_y,
            color="red",
            alpha=1.0,
            s=2,
            label="Incorrect Prediction",
        )
        plt.xlabel(f"X ({x_colname})")
        plt.ylabel(f"Y ({y_colname})")
        plt.title(
            f"Performance of numeric X ({x_colname}) predicting categorical Y ({y_colname})\n"
            f"{metric_name}: {metric_value:,.2f}",
        )
        plt.yticks(
            ticks=range(len(category_nums_map)), labels=list(category_nums_map.keys())
        )
        plt.legend()
    elif isinstance(x[0], str) and isinstance(y_pred[0], (int | float)):
        sns.boxplot(
            x=x,
            y=y_true,
            flierprops={
                "marker": "o",
                "markersize": 1,
                "alpha": 0.6,
            },
        )
        sns.stripplot(
            x=x,
            y=y_pred,
            color="red",
            jitter=True,
            label="Predicted",
            alpha=0.6,
        )
        plt.xlabel(f"X ({x_colname})")
        plt.ylabel(f"Y ({y_colname})")
        plt.title(
            f"Performance of categorical X ({x_colname}) predicting numeric Y ({y_colname})\n"
            f"{metric_name}: {metric_value:,.2f}"
        )
        plt.legend()
    elif isinstance(x[0], str) and isinstance(y_pred[0], str):
        sns.countplot(x=x, hue=y_true)
        plt.xlabel("X (Categorical)")
        plt.ylabel("Count of Y (Categorical)")
        x_preds = {}
        for x_i, y_pred_i in zip(x, y_pred):
            if x_i not in x_preds:
                x_preds[x_i] = y_pred_i
        plt.xlabel(f"X ({x_colname})")
        plt.ylabel(f"Y ({y_colname})")
        plt.title(
            f"Performance of categorical X ({x_colname}) predicted categorical Y ({y_colname})\n"
            f"{metric_name}: {metric_value:,.2f}\n"
            f"Predictions: {x_preds}"
        )

    plt.tight_layout()
    plt.show()


# data splitting #
df = df.collect()
df = df.to_pandas()  # I'm not happy about this
X = df.drop("price", axis=1)
y = df["price"].to_numpy()
df_train_valid, df_test, X_train_valid, X_test, y_train_valid, y_test = (
    train_test_split(df, X, y, test_size=0.1, random_state=69420)
)
df_train, df_valid, X_train, X_valid, y_train, y_valid = train_test_split(
    df_train_valid, X_train_valid, y_train_valid, test_size=0.5, random_state=42069
)
print(
    f"""
-- Summary of data splits -- 

Training:   {X_train.shape[0]:,} rows ({(100*X_train.shape[0]/df.shape[0]):,.1f}%)
Validation: {X_valid.shape[0]:,} rows ({(100*X_valid.shape[0]/df.shape[0]):,.1f}%)
Testing:    {X_test.shape[0]:,} rows ({(100*X_test.shape[0]/df.shape[0]):,.1f}%)
TOTAL:      {df.shape[0]:,} rows
"""
)

numeric_feature_colnames: set[str] = set()
categorical_feature_colnames: set[str] = set()
response_colname: str = "price"
for colname in X.columns:
    if X[colname].dtype.kind in "iufc":
        numeric_feature_colnames.add(colname)
    else:
        categorical_feature_colnames.add(colname)
print(
    f"""
Numeric features are: {", ".join(numeric_feature_colnames)}

Categorical features are: {", ".join(categorical_feature_colnames)}

Response is: {response_colname}
"""
)

# assess relationships between features (and response) using x2y metric (predictive power score) #
x2y_decision_tree_regressor_kwargs = {"max_depth": 5}
x2y_decision_tree_classifier_kwargs = {"max_depth": 5}
x2y_classification_pipeline = Pipeline(
    steps=[
        (
            "one_hot",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        (
            "classifier",
            DecisionTreeClassifier(**x2y_decision_tree_classifier_kwargs),
        ),
    ]
)
x2y_regression_pipeline = Pipeline(
    steps=[
        (
            "one_hot",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        (
            "regressor",
            DecisionTreeRegressor(**x2y_decision_tree_regressor_kwargs),
        ),
    ]
)
result_df_rows_numeric_y: list[pd.DataFrame] = []
result_df_rows_categorical_y: list[pd.DataFrame] = []
plot_y_names: list[str] = [
    # if predicted y appears in this list, then a prediction plot is displayed
    #   when that variable is predicted
    # pass an empty list to generate no plots
    "price",
    "carat",
    # "cut",
    "color",
    "clarity",
    # "depth_percent",
    # "table",
    # "length",
    # "width",
    # "depth",
]
# predict every numeric column #
for x_colname in tqdm(df_train.columns):
    for y_colname in ["price"] + list(numeric_feature_colnames):
        if x_colname == y_colname:
            continue
        x = df_train[[x_colname]]
        y = df_train[[y_colname]]
        x_outsample = df_valid[[x_colname]]
        y_outsample = df_valid[[y_colname]]
        baseline_preds_y_outsample = (
            # just predict the mean every time
            y[y_colname].mean()
            * np.ones((y_outsample.shape[0], 1))
        )
        pipeline = x2y_regression_pipeline
        pipeline.fit(x, y)
        model_preds_y_outsample = pipeline.predict(x_outsample)
        baseline_outsample_mae = mean_absolute_error(
            y_outsample, baseline_preds_y_outsample
        )
        model_outsample_mae = mean_absolute_error(y_outsample, model_preds_y_outsample)
        result_df_rows_numeric_y.append(
            pd.DataFrame(
                {
                    "x": x_colname,
                    "y": y_colname,
                    "metric": "Mean Absolute Error",
                    "metric_baseline": baseline_outsample_mae,
                    "metric_model": model_outsample_mae,
                    "metric_ratio": baseline_outsample_mae / model_outsample_mae,
                },
                index=[0],
            )
        )
        if y_colname in plot_y_names:
            univariate_prediction_plot(
                x=x_outsample[x_colname].tolist(),
                y_pred=model_preds_y_outsample.tolist(),
                y_true=y_outsample[y_colname].tolist(),
                x_colname=x_colname,
                y_colname=y_colname,
                metric_name="outsample Mean Absolute Error (MAE)",
                metric_value=float(model_outsample_mae),
            )
# predict every categorical column #
for x_colname in tqdm(df_train.columns):
    for y_colname in list(categorical_feature_colnames):
        if x_colname == y_colname:
            continue
        x = df_train[[x_colname]]
        y = df_train[[y_colname]]
        x_outsample = df_valid[[x_colname]]
        y_outsample = df_valid[[y_colname]]
        baseline_preds_y_outsample = (
            # just predict the most common label each time
            np.array([y[y_colname].mode()[0]] * y.shape[0])
        )
        pipeline = x2y_classification_pipeline
        pipeline.fit(x, y)
        model_preds_y_outsample = pipeline.predict(x_outsample)
        baseline_outsample_accuracy = accuracy_score(
            y_outsample,
            baseline_preds_y_outsample,
        )
        model_outsample_accuracy = accuracy_score(
            y_outsample,
            model_preds_y_outsample,
        )
        result_df_rows_categorical_y.append(
            pd.DataFrame(
                {
                    "x": x_colname,
                    "y": y_colname,
                    "metric": "Accuracy",
                    "metric_baseline": baseline_outsample_accuracy,
                    "metric_model": model_outsample_accuracy,
                    "metric_ratio": model_outsample_accuracy
                    / baseline_outsample_accuracy,
                },
                index=[0],
            )
        )
        if y_colname in plot_y_names:
            univariate_prediction_plot(
                x=x_outsample[x_colname].tolist(),
                y_pred=model_preds_y_outsample.tolist(),
                y_true=y_outsample[y_colname].tolist(),
                x_colname=x_colname,
                y_colname=y_colname,
                metric_name="outsample accuracy",
                metric_value=model_outsample_accuracy,
            )
# x2y heatmap for numeric y #
x2y_numeric_y_df = pd.concat(result_df_rows_numeric_y, axis=0)
heatmap_data = x2y_numeric_y_df.pivot(index="x", columns="y", values="metric_ratio")
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True)
plt.suptitle("x2y score (numeric Y variables only)")
plt.title("(how well does each X individually predict each Y)")
plt.ylabel("predictor (x)")
plt.xlabel("predicted (y)")
plt.tight_layout()
plt.show()
# x2y heatmap for categorical y #
x2y_categorical_y_df = pd.concat(result_df_rows_categorical_y, axis=0)
heatmap_data = x2y_categorical_y_df.pivot(index="x", columns="y", values="metric_ratio")
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True)
plt.suptitle("x2y score (categorical Y variables only)")
plt.title("(how well does each X individually predict each Y)")
plt.ylabel("predictor (x)")
plt.xlabel("predicted (y)")
plt.tight_layout()
plt.show()


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
