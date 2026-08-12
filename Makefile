# Convenience targets. Use `make help` to list them.
PYTHON ?= python
PORT   ?= 8000

.PHONY: help install data features train promote serve dashboard test lint format clean mlflow-ui docker-build docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package + dev deps in editable mode
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

data: ## Generate synthetic raw data and run ingestion -> processed splits
	$(PYTHON) -m credit_risk.data.synthetic
	$(PYTHON) -m credit_risk.data.ingestion

features: ## (Optional) Materialise feature-store snapshots
	$(PYTHON) -c "from credit_risk.features.store import snapshot; snapshot()"

train: ## Train candidate models, log to MLflow, register champion
	$(PYTHON) -m credit_risk.models.train

promote: ## Promote the latest model version to Production
	$(PYTHON) -c "from credit_risk.models.registry import promote_latest_to_production; promote_latest_to_production()"

serve: ## Run the FastAPI scoring service
	$(PYTHON) -m credit_risk.serving.app

dashboard: ## Run the Streamlit business dashboard
	streamlit run dashboard/app.py

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint with ruff
	$(PYTHON) -m ruff check src tests

format: ## Auto-format with ruff
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

mlflow-ui: ## Open the MLflow tracking UI on :5000
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

clean: ## Remove generated data, models and caches
	rm -rf data/raw/* data/processed/* data/drift_reference/* mlruns mlflow.db mlflow.db-* .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

docker-build: ## Build the serving image
	docker build -t credit-risk-mlops:latest .

docker-up: ## Run the full stack (API + MLflow) via docker compose
	docker compose up -d

docker-down: ## Stop the docker compose stack
	docker compose down
