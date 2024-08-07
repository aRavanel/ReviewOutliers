from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from src.tasks.outliers_and_shift import shift_detection
from src.config import logger
from src.api.utils import BatchDetectionRequest, convert_requests_to_dataframe

# ==========================================================================
# module variables and classes
# ==========================================================================

shift_router = APIRouter()


# Pydantic Schema for Distribution Shift
class DistributionShiftResponse(BaseModel):
    score: float


class BatchDistributionShiftResponse(BaseModel):
    results: List[DistributionShiftResponse]


# ==========================================================================
# Exported functions
# ==========================================================================


@shift_router.post("/distribution_shift", response_model=BatchDistributionShiftResponse)
def distribution_shift(request: BatchDetectionRequest) -> BatchDistributionShiftResponse:
    """
    Calculates the distribution shift scores for a batch of data.

    Args:
        request (BatchDetectionRequest): The input data in the form of a batch request.

    Returns:
        BatchDistributionShiftResponse: The distribution shift scores for each data point in the batch.
    """
    # Log the API call
    logger.debug("calling distribution_shift")

    # Convert the requests to a dataframe
    df = convert_requests_to_dataframe(request.requests)

    # Calculate the distribution shift scores
    list_score = shift_detection(df, training=False)

    # Create a list of DistributionShiftResponse objects
    results = [DistributionShiftResponse(score=score) for score in list_score]

    # Return the response
    return BatchDistributionShiftResponse(results=results)
