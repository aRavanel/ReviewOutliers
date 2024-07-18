.PHONY: install test lint format

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