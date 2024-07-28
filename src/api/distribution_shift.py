from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.utils.preprocessing.preprocessing_cleaning import preprocess_data, clean_enrich_reviews, clean_enrich_metadata
from src.utils.preprocessing.preprocessing_encoding import distribution_shift_function

# ==========================================================================
# Shema and module variables
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

# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/distribution_shift", response_model=BatchDistributionShiftResponse)
def distribution_shift(request: BatchDistributionShiftRequest):
    # Construct a DataFrame from the list of requests
    df = pd.DataFrame([req.dict() for req in request.requests])

    # Preprocess the data
    df = clean_enrich_reviews(df)

    # Apply the distribution shift function to each processed feature set
    df["shift_score"] = df["processed_features"].apply(distribution_shift_function)

    # Convert the results to a list of DistributionShiftResponse
    responses = [DistributionShiftResponse(shift_score=row["shift_score"]) for index, row in df.iterrows()]

    return BatchDistributionShiftResponse(responses=responses)
