# Use the official slim Python image from the Python 3.12.2 version
FROM python:3.12.2-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock ./

# Install Poetry, a dependency management tool
RUN pip install poetry

# Install the project dependencies specified in pyproject.toml, excluding dev dependencies
RUN poetry install --no-dev

# Copy the source code to the container's /app/src directory
COPY src ./src

# Copy the data models to the container's /app/data/models directory
COPY data/models ./data/models

# Copy any other necessary directories
COPY logs ./logs

# Define the command to run the application using Uvicorn via Poetry
CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
