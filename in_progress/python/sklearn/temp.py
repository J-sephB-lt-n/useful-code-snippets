from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_per_subgroup(
    subgroup_column: pd.Series,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric_func: Callable,
) -> None:
    """
    Plots a (sorted) matplotlib barchart showing the value of a metric within each
    label of a categorical feature

    Args:
        subgroup_column (pandas.Series): The categorical feature which defines the subgroup
        y_pred (numpy.ndarray): Predicted outcome (y) values
        y_true (numpy.ndarray): True outcome (y) values
        metric_func (Callable): A function using `y_pred` and `y_true` as input and
                                producing a single number
                                e.g. sklearn.metric.mean_squared_error

    Example:
        >>> from sklearn.metrics import mean_absolute_percentage_error
        >>> metric_per_subgroup(
        ...     subgroup_column=pd.Series( ["a","b","a","b","a"] ),
        ...     y_pred=np.array( [2,3,4,3,1] ),
        ...     y_true=np.array( [1,3,5,4,2] ),
        ...     metric_func=mean_absolute_percentage_error
        ... )
    """
    metrics_by_group = {}
    for label in subgroup_column.unique():
        mask = subgroup_column == label
        metrics_by_group[label] = metric_func(y_true[mask], y_pred[mask])
    sorted_groups = sorted(metrics_by_group.items(), key=lambda x: x[1])
    labels, metrics = zip(*sorted_groups)
    plt.figure(figsize=(10, 7))
    plt.barh(labels, metrics)
    plt.xlabel(f"Metric: {metric_func.__name__}")
    plt.ylabel("Subgroup")
    plt.title(f"{metric_func.__name__} per Subgroup")
    plt.tight_layout()
    plt.show()
