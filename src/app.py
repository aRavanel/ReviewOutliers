from fastapi import FastAPI

# Import the routers for different endpoints
from src.api import reviews, outlier_detection, distribution_shift

# Create a FastAPI application
app = FastAPI()

# Include the routers for different endpoints with appropriate prefixes
app.include_router(reviews.router, prefix="/api/reviews")  # For reviews endpoints
app.include_router(outlier_detection.router, prefix="/api")  # For outlier detection endpoints
app.include_router(distribution_shift.router, prefix="/api")  # For distribution shift scoring endpoints


# Define the root endpoint
@app.get("/")
def read_root():
    """
    Root endpoint for the Outlier Detection and Distribution Shift API.
    Returns a welcome message.
    """
    return {"message": "Welcome 👋 to the Outlier Detection and Distribution Shift API"}
