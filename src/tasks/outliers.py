import os
import pickle
from typing import Tuple
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.suod import SUOD
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
import pandas as pd
import numpy as np

# module imports
from src.config import logger


# ==========================================================================
# Utils functions and module
# ==========================================================================
from src.config import MODEL_NAME_OUTLIER, MODEL_PATH_OUTLIER
from src.utils.preprocessing.preprocessing import preprocess_data


def _save_model_outlier(model) -> None:
    """Save the model to a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        logger.error(f"Error saving the model: {e}")
        logger.info(f"Current Directory: {os.getcwd()}")


def _load_model_outlier():
    """Load the model from a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "rb") as file:
            model = pickle.load(file)
        return model
    except IOError as e:
        logger.error(f"Error loading the model: {e}")
        raise e


# ==========================================================================
# Exported functions
# ==========================================================================
def outlier_prediction(
    df: pd.DataFrame, training: bool = True, outlier_on_score: bool = True, contamination: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict outliers and compute scores for the test set.
    Notes :
    - Train the Isolation Forest model on the training set and optionally save it.
    - scores > 0   ->   outlier
    - contamination of 0.1 -> expect 10% of outliers
    """
    logger.debug("calling outlier_prediction")

    # outlier detection
    if training:
        match MODEL_NAME_OUTLIER:

            case "isolation_forest":
                model = IForest(n_estimators=100, max_samples="auto", contamination=contamination, random_state=42)

            case "one-class-svm":
                model = OCSVM(kernel="rbf", contamination=contamination)

            case "ensemble":
                # initialized a group of outlier detectors for acceleration
                detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20), COPOD(), IForest(n_estimators=100)]

                # decide the number of parallel process, and the combination method
                model = SUOD(base_estimators=detector_list, n_jobs=2, combination="average", verbose=False)

            case _:
                raise ValueError(f"Invalid model name: {MODEL_NAME_OUTLIER}")

        model.fit(df)
        _save_model_outlier(model)

    else:
        model = _load_model_outlier()

    scores = np.array(model.decision_function(df))  # list of scores

    if outlier_on_score:
        # Calculate IQR and determine a threshold for outliers
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        threshold_lower = Q1 - 1.5 * IQR
        threshold_upper = Q3 + 1.5 * IQR
        outliers = (scores < threshold_lower) | (scores > threshold_upper)
    else:
        # fixed contamination -> too rigid
        outliers = model.predict(df)  # list of 0 (inliner) and 1 (outlier)

    return np.array(outliers), scores


def outlier_detection(df: pd.DataFrame, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the Isolation Forest model on the training set and optionally save it.
    scores > 0   ->   outlier
    Predict outliers and compute scores for the test set.
    """
    logger.debug("calling outlier_detection")

    # clean, enrich, encode the data. TODO : have one specific preprocessing per model
    df = preprocess_data(df, training=False)

    # do the prediction
    outliers, scores = outlier_prediction(df, training)

    return outliers, scores
