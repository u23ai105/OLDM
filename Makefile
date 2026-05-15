.PHONY: install install-gpu install-cpu dev test lint format \
       run-api run-worker run-pipeline download-weights \
       docker-build docker-up clean help

PYTHON = python3
PIP = pip3
APP_MODULE = src.api.main:app

help:
	@echo ""
	@echo "  AI Cinematic Restoration Pipeline"
	@echo "  ================================="
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install base dependencies"
	@echo "    make install-gpu      Install with CUDA/GPU support"
	@echo "    make install-cpu      Install CPU-only (local dev)"
	@echo "    make dev              Install dev tools + pre-commit hooks"
	@echo "    make download-weights Download all model weights (~2GB)"
	@echo ""
	@echo "  Run:"
	@echo "    make run-api          Start FastAPI server"
	@echo "    make run-worker       Start Prefect worker"
	@echo "    make run-pipeline V=<path>  Run pipeline on a video"
	@echo ""
	@echo "  Quality:"
	@echo "    make test             Run test suite"
	@echo "    make lint             Check code style (flake8, mypy)"
	@echo "    make format           Auto-format code (black, isort)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build     Build Docker images"
	@echo "    make docker-up        Start services with Docker Compose"
	@echo ""
	@echo "    make clean            Remove caches and temp files"
	@echo ""

# ── Setup ──────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt

install-gpu:
	$(PIP) install -r requirements-gpu.txt

install-cpu:
	$(PIP) install -r requirements-cpu.txt

dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

download-weights:
	$(PYTHON) scripts/download_weights.py
	@echo "✓ All model weights downloaded to ./weights/"

# ── Quality ────────────────────────────────────────────────

test:
	pytest tests/ --cov=src -v

lint:
	flake8 src/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# ── Run ────────────────────────────────────────────────────

run-api:
	uvicorn $(APP_MODULE) --host 0.0.0.0 --port 8000 --reload

run-worker:
	prefect worker start --pool "restoration-pool"

run-pipeline:
ifndef V
	@echo "Usage: make run-pipeline V=data/raw/movie.mp4"
	@exit 1
endif
	$(PYTHON) -m src.pipeline.flows $(V)

# ── Docker ─────────────────────────────────────────────────

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

# ── Cleanup ────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@echo "✓ Cleaned"
