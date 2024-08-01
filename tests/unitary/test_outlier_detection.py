import os

# module imports
from src.utils.io import load_dataframe
from src.config import BASE_PATH_DATA
from src.config import logger
from src.api.outlier_detection import create_batch_outlier_request
from src.tasks.outliers import outlier_detection

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
    outliers, scores = outlier_detection(df_test, training=False)

    assert outliers.shape[0] == n_data
    assert scores.shape[0] == n_data
