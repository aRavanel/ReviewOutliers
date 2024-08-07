import os

# module imports
from src.utils.io.io import load_dataframe
from src.config import BASE_PATH_DATA
from src.tasks.outliers_and_shift import outlier_prediction

# ==========================================================================
# Module variables
# ==========================================================================
n_data = 2
df_test = load_dataframe(os.path.join(BASE_PATH_DATA, "processed", "test.parquet"))
df_test = df_test.sample(n_data)  # take some samples

# ==========================================================================
# Tests
# ==========================================================================
# format the data for a request


def test_outlier_detection_inference() -> None:
    """
    Basic test to see if outlier detection runs
    """
    outliers, scores = outlier_prediction(df_test, training=False)

    assert outliers.shape[0] == n_data
    assert scores.shape[0] == n_data
