import os
import logging
import logging.config

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
current_file_path = os.path.abspath(__file__)
PATH_SRC = os.path.dirname(current_file_path)
PATH_PROJECT = os.path.dirname(PATH_SRC)

BASE_PATH_MODEL = os.path.join(PATH_PROJECT, "data", "models")
BASE_PATH_DATA = os.path.join(PATH_PROJECT, "data")

MODEL_PATH_OUTLIER = os.path.join(BASE_PATH_MODEL, FILENAME_OUTLIER)
MODEL_PATH_STANDARDIZER = os.path.join(BASE_PATH_MODEL, FILENAME_STANDARDIZER)

# ==========================================================================
# logging
# ==========================================================================

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": "app.log",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
