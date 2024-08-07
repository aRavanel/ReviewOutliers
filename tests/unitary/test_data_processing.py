import os

# module imports
from src.utils.io.io import load_dataframe
from src.config import BASE_PATH_DATA
from src.utils.preprocessing.preprocessing import preprocess_data

# ==========================================================================
# Module variables
# ==========================================================================
n_data = 2
df = load_dataframe(os.path.join(BASE_PATH_DATA, "processed", "test.parquet"))
df = df.sample(n_data)  # take some samples


# ==========================================================================
# Tests
# ==========================================================================
def test_data_processing() -> None:

    df_processed = preprocess_data(df, training=False)

    assert df_processed.shape[0] <= df.shape[0]  # less samples
    assert df_processed.shape[1] >= df.shape[0]  # more features
