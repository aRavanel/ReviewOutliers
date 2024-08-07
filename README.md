# Outlier and distribution shift

Identify outliers and shift. Can be used to :
- Detect incorrect data
- Detect extreme cases
  - interesting / hard cases
  - fraud
  - ...

## Main files and folders in the repo

Folders :

- `data/`: contains downloaded and processed data
- `src/`: source code
- `notebooks/`: for dowloading, processing, and launching the services
  - `00_data_download.ipynb`: download wanted dataset
  - `01_data_merge_and_splitting.ipynb`: merge and choose splitting method
  - `02_data_exploration.ipynb`: automatic and manual exploration
  - `03_clean_enrich_encode.ipynb`: enrich and encode data. save the encoders
  - `04_outlier_and_shift.ipynb`: run and analyze some data
  - `05_model_serving.ipynb`: launch the services (uvicorn or docker)

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
    "python.linting.flake8Enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter",
    "notebook.defaultFormatter": "ms-python.flake8",
    "flake8.args": ["--max-line-length=128"],
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoSearchPaths": true,
    "python.analysis.useLibraryCodeForTypes": true,
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
- `POST /api/detect_outliers`: Detects if a given sample is an outlier.
- `POST /api/distribution_shift`: Computes the distribution shift score

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

- [x] Dataset selection
- [x] Outlier detection
- [x] Distribution shift scoring
- [x] Model serving
- [x] Documentation


# Resources : 

- dataset page: https://amazon-reviews-2023.github.io/main.html
- HF page: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- github: https://github.com/hyp1231/AmazonReviews2023
- 