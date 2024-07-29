import os
import pickle
from typing import List, Tuple
from pyod.models.iforest import IForest
import pandas as pd
import numpy as np


# ==========================================================================
# Utils functions and module
# ==========================================================================
from src.config import MODEL_NAME_OUTLIER, MODEL_PATH_OUTLIER


def save_model_outlier(model) -> None:
    """Save the model to a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        print(f"Error saving the model: {e}")
        print(f"Current Directory: {os.getcwd()}")


def load_model_outlier():
    """Load the model from a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "rb") as file:
            model = pickle.load(file)
        return model
    except IOError as e:
        print(f"Error loading the model: {e}")
        raise e


# ==========================================================================
# Exported functions
# ==========================================================================


def outlier_detection(df: pd.DataFrame, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the Isolation Forest model on the training set and optionally save it.
     scores > 0   ->   outlier
    Predict outliers and compute scores for the test set.
    """

    if training:
        # create the model
        match MODEL_NAME_OUTLIER:
            case "isolation_forest":
                model = IForest(contamination=0.1, random_state=42)  # 0.1 -> expect 10% of outliers
            case _:
                raise ValueError(f"Invalid model name: {MODEL_NAME_OUTLIER}")

        # train the model
        model.fit(df)
        save_model_outlier(model)

    else:
        # load the model
        model = load_model_outlier()

    # run the prediction
    outliers = model.predict(df)  # list of 0 (inliner) and 1 (outlier)
    scores = model.decision_function(df)  # list of scores

    return np.array(outliers), np.array(scores)
