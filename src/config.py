import os

# ==========================================================================
# User defined
# ==========================================================================

# text embeddings model from sentence transformers
MODEL_NAME_EMBEDDINGS = "all-MiniLM-L6-v2"  # dim : 384, max_len : 256

# outlier model name from pyod
MODEL_NAME_OUTLIER = "isolation_forest"

# pickle names
FILENAME_OUTLIER = "model_outlier.pkl"
FILENAME_STANDARDIZER = "standardizer.pkl"

# ==========================================================================
# paths variables
# ==========================================================================

# Paths
BASE_PATH_MODEL = os.path.join("data", "models")
BASE_PATH_DATA = os.path.join("data")
MODEL_PATH_OUTLIER = os.path.join(BASE_PATH_MODEL, FILENAME_OUTLIER)
MODEL_PATH_STANDARDIZER = os.path.join(BASE_PATH_MODEL, FILENAME_STANDARDIZER)
