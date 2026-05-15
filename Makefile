.PHONY: install dev test lint format run-api run-worker run-pipeline docker-build docker-up clean help

PYTHON = python3
PIP = pip3
APP_MODULE = src.api.main:app

help:
	@echo "Usage:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies"
	@echo "  make test           Run tests"
	@echo "  make lint           Check code style (flake8, mypy)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make run-api        Run FastAPI server"
	@echo "  make run-worker     Run Prefect worker"
	@echo "  make run-pipeline   Trigger a restoration pipeline manually"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start project with Docker Compose"
	@echo "  make clean          Clean temporary files"

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ --cov=src

lint:
	flake8 src/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

run-api:
	uvicorn $(APP_MODULE) --host 0.0.0.0 --port 8000 --reload

run-worker:
	prefect worker start --pool "restoration-pool"

run-pipeline:
	$(PYTHON) -m src.pipeline.flows

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
