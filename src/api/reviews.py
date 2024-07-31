from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# module imports
from logger_config import logger


# ==========================================================================
# Shema and module variables
# ==========================================================================

router = APIRouter()
reviews_db = []  # In-memory storage for the example


# Example data model
class Review(BaseModel):
    """
    Review data model.
    """

    user_id: str
    review_text: str
    rating: int


# ==========================================================================
# Exported functions
# ==========================================================================


@router.get("/")
async def read_root():
    """
    Root endpoint to return a welcome message.
    """
    logger.debug("calling read_root")
    return {"message": "Welcome to the Review Outliers API"}


@router.get("/reviews", response_model=List[Review])
async def get_reviews():
    """
    Get all reviews.
    """
    logger.debug("calling get_reviews")
    return reviews_db


@router.post("/reviews", response_model=Review)
async def create_review(review: Review):
    """
    Create a new review.
    """
    reviews_db.append(review)
    return review


@router.get("/reviews/{user_id}", response_model=Review)
async def get_review_by_user(user_id: str):
    """
    Get a review by user ID.
    """
    for review in reviews_db:
        if review.user_id == user_id:
            return review
    raise HTTPException(status_code=404, detail="Review not found")


@router.put("/reviews/{user_id}", response_model=Review)
async def update_review(user_id: str, updated_review: Review):
    """
    Update a review by user ID.
    """
    for index, review in enumerate(reviews_db):
        if review.user_id == user_id:
            reviews_db[index] = updated_review
            return updated_review
    raise HTTPException(status_code=404, detail="Review not found")


@router.delete("/reviews/{user_id}", response_model=Review)
async def delete_review(user_id: str):
    """
    Delete a review by user ID.
    """
    for index, review in enumerate(reviews_db):
        if review.user_id == user_id:
            return reviews_db.pop(index)
    raise HTTPException(status_code=404, detail="Review not found")
