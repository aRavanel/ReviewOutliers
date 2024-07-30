from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
from src.utils.preprocessing.preprocessing import preprocess_data
from src.tasks.distribution_shift import distribution_shift_scoring

# ==========================================================================
# Schema and module variables
# ==========================================================================


# Define request and response models
class DistributionShiftRequest(BaseModel):
    text: str
    features: List[float]


class BatchDistributionShiftRequest(BaseModel):
    requests: List[DistributionShiftRequest]


class DistributionShiftResponse(BaseModel):
    shift_score: float


class BatchDistributionShiftResponse(BaseModel):
    responses: List[DistributionShiftResponse]


router = APIRouter()

# Load metadata dataframe
df_metadata = pd.read_parquet(os.path.join("..", "data", "processed", "metadata.parquet"))

# load the model

# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/distribution_shift", response_model=BatchDistributionShiftResponse)
def distribution_shift(request: BatchDistributionShiftRequest):
    # Construct a DataFrame from the list of requests
    df_merged = pd.DataFrame([req.dict() for req in request.requests])

    # Preprocess the data
    df = preprocess_data(df_merged, training=False)

    # Apply the distribution shift function to each processed feature set
    df["shift_score"] = df["processed_features"].apply(distribution_shift_scoring)

    # Convert the results to a list of DistributionShiftResponse
    responses = [DistributionShiftResponse(shift_score=row["shift_score"]) for index, row in df.iterrows()]

    return BatchDistributionShiftResponse(responses=responses)
