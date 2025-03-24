# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES (optional)
# ---------------------------------------------------------
-include .env
ifneq ("$(wildcard .env)","")
  export $(shell sed 's/=.*//' .env)
endif


# ---------------------------------------------------------
# HELP / USAGE (Default Target)
# ---------------------------------------------------------
.PHONY: help
help:
	@echo "===================================================="
	@echo "                Available Make Targets"
	@echo "===================================================="
	@echo "make setup               - Install Poetry dependencies"
	@echo "make download-data       - Download raw data files via script"
	@echo "make run-pipeline        - Run the image filter pipeline (Ray + Dask)"
	@echo "make clean-data          - Clean processed and interim data folders"
	@echo "make test                - Run unit tests (pytest)"
	@echo "make format              - Auto-format code with Black"
	@echo "make ci-format           - Check code format (CI/CD)"
	@echo "===================================================="

.DEFAULT_GOAL := help


# ---------------------------------------------------------
# DATA TASKS
# ---------------------------------------------------------
.PHONY: download-data
# Downloads required raw parquet files from Hugging Face
# Ensures files exist locally as defined in config.py

download-data:
	@echo "===================================================="
	@echo "Running data download script..."
	@echo "===================================================="
	python scripts/download_data.py
	@echo "✅ Data download completed!"

.PHONY: clean-data
clean-data:
	@echo "===================================================="
	@echo "Cleaning interim and processed data directories..."
	@echo "===================================================="
	rm -rf image_filter_pipeline/data/interim/*.parquet
	rm -rf image_filter_pipeline/data/processed/*.parquet
	@echo "🧹 Data directories cleaned!"


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------
.PHONY: run-pipeline
run-pipeline:
	@echo "===================================================="
	@echo "Running image filter pipeline with Dask & Ray..."
	@echo "===================================================="
	poetry install --no-cache
	python main.py
	@echo "✅ Pipeline execution completed!"


# ---------------------------------------------------------
# TESTS / FORMATTING
# ---------------------------------------------------------
.PHONY: test
test:
	@echo "===================================================="
	@echo "Running unit tests with pytest..."
	@echo "===================================================="
	PYTHONPATH=. pytest tests/
	@echo "✅ Tests completed!"

.PHONY: format
format:
	@echo "===================================================="
	@echo "Auto-formatting code using Black..."
	@echo "===================================================="
	black .
	@echo "✅ Formatting completed!"

.PHONY: ci-format
ci-format:
	@echo "===================================================="
	@echo "Running Black in check mode for CI..."
	@echo "===================================================="
	black --check .

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
.PHONY: setup
setup:
	@echo "===================================================="
	@echo "Installing project dependencies with Poetry..."
	@echo "===================================================="
	poetry install --no-cache
	@echo "✅ Dependencies installed!"