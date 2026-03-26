.PHONY: install format lint typecheck test all clean

install:
	pip install -e ".[dev]"
	pre-commit install

format:
	python -m ruff format source tests

lint:
	python -m ruff check source tests --fix

typecheck:
	python -m mypy

test:
	python -m pytest tests/

all: format lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
