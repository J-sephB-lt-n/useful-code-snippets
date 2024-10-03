"""
TAGS: optim|optimisation|optimization|hyperparameter|hyperparameter optimisation|optuna
DESCRIPTION: TODO
REQUIREMENTS: pip install 'scikit-learn==1.5.2' 'optuna==4.0.0'
"""

import optuna
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()
X_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y = housing_data.target
