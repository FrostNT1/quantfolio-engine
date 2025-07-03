#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = quantfolio-engine
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	pip install -e ".[dev]"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 quantfolio_engine
	isort --check --diff quantfolio_engine
	black --check quantfolio_engine

## Format source code with black
.PHONY: format
format:
	isort quantfolio_engine
	black quantfolio_engine

## Run tests
.PHONY: test
test:
	python -m pytest tests -v

## Run tests with coverage
.PHONY: test-cov
test-cov:
	python -m pytest tests --cov=quantfolio_engine --cov-report=html --cov-report=term

## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://bucket-name/data/ \
		data/

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://bucket-name/data

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Load data pipeline
.PHONY: data
data: requirements
	quantfolio fetch-data

## Fetch specific data types
.PHONY: data-returns
data-returns: requirements
	quantfolio fetch-data --type returns

.PHONY: data-macro
data-macro: requirements
	quantfolio fetch-data --type macro

.PHONY: data-sentiment
data-sentiment: requirements
	quantfolio fetch-data --type sentiment

## List available data
.PHONY: list-data
list-data: requirements
	quantfolio list-data

## Show system status
.PHONY: status
status: requirements
	quantfolio status

## Validate data quality
.PHONY: validate-data
validate-data: requirements
	quantfolio validate-data

## Generate factor timing signals
.PHONY: signals
signals: data
	$(PYTHON_INTERPRETER) -m quantfolio_engine.signals.factor_timing

## Run portfolio optimization
.PHONY: optimize
optimize: signals
	$(PYTHON_INTERPRETER) -c "from quantfolio_engine.optimizer.black_litterman import BlackLittermanOptimizer; print('Optimizer ready')"

## Run risk attribution
.PHONY: attribution
attribution: optimize
	$(PYTHON_INTERPRETER) -c "from quantfolio_engine.attribution.risk_attribution import RiskAttribution; print('Attribution ready')"

## Launch dashboard
.PHONY: dashboard
dashboard: requirements
	streamlit run quantfolio_engine/dashboard/app.py

## Run full pipeline
.PHONY: pipeline
pipeline: data signals optimize attribution
	@echo "Full pipeline completed"

## Validate configuration
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -c "from quantfolio_engine.config import validate_config; validate_config()"

## Create environment file template
.PHONY: env-template
env-template:
	@echo "Creating .env template..."
	@echo "# QuantFolio Engine Configuration" > .env.example
	@echo "FRED_API_KEY=your_fred_api_key_here" >> .env.example
	@echo "NEWS_API_KEY=your_news_api_key_here" >> .env.example
	@echo "ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here" >> .env.example
	@echo ".env.example created. Copy to .env and add your API keys."

## Install pre-commit hooks
.PHONY: pre-commit
pre-commit: requirements
	pre-commit install

## Run pre-commit on all files
.PHONY: pre-commit-all
pre-commit-all: pre-commit
	pre-commit run --all-files

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) quantfolio_engine/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
