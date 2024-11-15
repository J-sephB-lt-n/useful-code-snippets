"""
TAGS: file|joblib|load|model|model persistence|persist|persistence|pickle|save|scikit-learn|sklearn
DESCRIPTION: Code (using joblib) for saving a trained model to a single file, and loading it again at a later stage
REQUIREMENTS: pip install joblib scikit-learn
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

# save model #
joblib.dump(model, "random_forest_model.joblib")

# load model #
loaded_model = joblib.load("random_forest_model.joblib")
