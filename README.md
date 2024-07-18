# Project

Identify outliers and shift

# Main files and folders in the repo

Folders :
`data/`: contains xxx
`src/`: source code of utils functions

Notebooks:

- `01_data_exploration.ipynb`: xxx
- `02_feature_engineering.ipynb`: xxx
- `03_outlier_detection.ipynb`: xxx
- `04_distribution_shift_scoring.ipynb`: xxx

# Set-up

## Python environment

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

## VS Code environement

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

## Clone the repo

```bash
git clone https://github.com/abwaab/ReviewOutliers.git
```

# Run the app

## Local Development and Testing

Run the FastAPI app using Uvicorn:

```bash
poetry run uvicorn src.app:app --reload
```

# Contribute

This project uses GitHub Actions for Continuous Integration (CI). The CI workflow is defined in .github/workflows/ci.yml. The workflow installs dependencies, runs tests, and performs linting and formatting checks on every push and pull request.

To ensure consistency and code quality, please run the following checks locally before committing your changes:

- Run Tests: `poetry run pytest`
- Run Linting: `poetry run ruff .`
- Run Formatting Check: `poetry run black --check .`

## Production Deployment and Consistent Environments

Build and run with Docker:

```bash
docker build -t ReviewOutliers .
docker run -p 8000:8000 ReviewOutliers
```

# Functionalities taken into account

Functionalities:

- [ ] Dataset selection
- [ ] Outlier detection
- [ ] Distribution shift scoring
- [ ] Model serving
- [ ] Documentation
