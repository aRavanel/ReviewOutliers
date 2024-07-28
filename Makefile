.PHONY: install test lint format run_fast build_docker run_docker

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run black .

full:
	make lint
	make format

run_fast:
	poetry run uvicorn src.app:app --reload

build_docker:
	# Build the Docker image
	docker build -t outlier-detection-api .

run_docker:
	# Run the Docker container
	docker run -d --name outlier-detection-api -p 8000:8000 outlier-detection-api
