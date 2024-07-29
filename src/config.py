# load model from sentence transformers
MODEL_NAME_EMBEDDINGS = (
    "all-MiniLM-L6-v2"  # dim : 384, max_len : 256  - A good balance between performance and dimensionality
)

# outlier model name
MODEL_NAME = "isolation_forest"  #

# pickle names
FILENAME_OUTLIER = "outlier_model.pickle"
