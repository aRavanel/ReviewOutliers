# Outlier and distribution shift

Identify outliers and shift

## Main files and folders in the repo

Folders :

- `data/`: contains xxx
- `src/`: source code of utils functions

Notebooks:

- `01_data_exploration.ipynb`: xxx
- `02_feature_engineering.ipynb`: xxx
- `03_outlier_detection.ipynb`: xxx
- `04_distribution_shift_scoring.ipynb`: xxx

## Set-up

### Python environment

Install:

- Initialize the project: `poetry install`
- Activate the virtual environment : `poetry shell`

Or :

```bash
pyenv versions
pyenv local 3.12.2
poetry env use $(pyenv which python)
poetry install
poetry shell
```

Note : if there are any issues with the above commands, pleaser refer to the additionnal documentation into `docs/`

### VS Code environement

```bash
mkdir -p .vscode
cat <<EOL >> .vscode/settings.json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": false,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true
}
EOL
```

### Clone the repo

```bash
git clone https://github.com/aRavanel/ReviewOutliers.git
```

## Run the app

### Local Development and Testing

Run the FastAPI app using Uvicorn `make run_fast` or:

```bash
poetry run uvicorn src.app:app --reload
```

### Run the app in production using docker

```bash
make build_docker
make run_docker
```

### API Endpoints

The API available at: `http://localhost:8000`.
Documentation at: `http://localhost:8000/docs`.

- `GET /api/reviews`: Retrieves all reviews.
- `POST /api/reviews`: Creates a new review.
- `GET /api/reviews/{user_id}`: Retrieves a review by user ID.
- `PUT /api/reviews/{user_id}`: Updates a review by user ID.
- `DELETE /api/reviews/{user_id}`: Deletes a review by user ID.
- `POST /api/detect_outliers`: Detects if a given sample is an outlier.
- `POST /api/distribution_shift`: Computes the distribution shift score for a given sample.

### make a request

1. Using shell:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/detect_outliers' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "string",
  "features": [
    0
  ]
}'
```
2. Using HTTPie : `http POST http://localhost:8000/api/detect_outliers text="string" features:=[0]`
3. Using python requests:
```python
import requests
url = "http://localhost:8000/api/detect_outliers"
payload = {"text": "string", "features": [0]}
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
response = requests.post(url, json=payload, headers=headers)
print(response.json())
```
4. Other tools : Postman, swagger docs, ...

## Contribute

This project uses GitHub Actions for Continuous Integration (CI). The CI workflow is defined in .github/workflows/ci.yml. The workflow installs dependencies, runs tests, and performs linting and formatting checks on every push and pull request.

To ensure consistency and code quality, please run the following checks locally before committing your changes:

- Run Tests: `poetry run pytest`
- Run Linting: `poetry run ruff .`
- Run Formatting Check: `poetry run black --check .`

Or use Make :

- `make test`
- `make lint`
- `make format`
- `make full`

## Functionalities taken into account

Functionalities:

- [ ] Dataset selection
- [ ] Outlier detection
- [ ] Distribution shift scoring
- [ ] Model serving
- [ ] Documentation


# Resources : 

- dataset page: https://amazon-reviews-2023.github.io/main.html
- HF page: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- github: https://github.com/hyp1231/AmazonReviews2023
- 