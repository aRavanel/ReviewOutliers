import os
import logging
import logging.config

# ==========================================================================
# User defined
# ==========================================================================

# text embeddings model from sentence transformers
# - "dunzhang/stella_en_1.5B_v5"  # ranked 2th
# - "dunzhang/stella_en_400M_v5"  # ranked 6th
# - "Alibaba-NLP/gte-large-en-v1.5"  # ranked 21
# - "BAAI/bge-large-en-v1.5"
# - "Alibaba-NLP/gte-base-en-v1.5"  # good compromise
# - "BAAI/bge-small-en-v1.5"
# - "all-MiniLM-L6-v2"  # ranked 117 dim : 384, max_len : 256
MODEL_NAME_EMBEDDINGS = "Alibaba-NLP/gte-base-en-v1.5"  # ranked 21


# outlier model name from pyod
MODEL_NAME_OUTLIER = "one-class-svm"  # "one-class-svm", "isolation_forest", "ensemble"

# pickle names
FILENAME_OUTLIER = "model_outlier.pkl"
FILENAME_STANDARDIZER = "standardizer.pkl"

# logging level
LOGGING_LEVEL = "DEBUG"  # DEBUG. INFO, WARNING, ERROR

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

LOG_DIR = os.path.join(PATH_PROJECT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

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
            "level": LOGGING_LEVEL,
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": LOGGING_LEVEL,
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": os.path.join(
                LOG_DIR, "app.log"
            ),  # else if relative path can cause issue if launcher from other folder
            "mode": "w",  # Open the log file in write mode to overwrite it each time
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": LOGGING_LEVEL,
            "propagate": True,
        },
        "matplotlib.font_manager": {  # specific logger
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
