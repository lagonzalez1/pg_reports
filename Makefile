PYTHON := python3
PYTEST := pytest

TEST_DIR_DA := Disability_analysis/test
TEST_DA := $(TEST_DIR_DA)/test_disability_analysis.py

TEST_DIR_AA := Assessment_analysis/test
TEST_AA := $(TEST_DIR_AA)/test_assessment_analysis.py


.PHONY: help test lint clean venv

help:
	@echo "Available targets:"
	@echo "  make test     - run unit tests with pytest"
	@echo "  make lint     - run flake8 lint checks"
	@echo "  make clean    - remove Python cache/__pycache__ files"
	@echo "  make venv     - create virtual environment"

test:
	@echo "Running test in $(TEST_DA), $(TEST_AA)"
	@$(PYTHON) -m pip install -q pytest
	@$(PYTHON) -m $(PYTEST) $(TEST_DA) -v
	@$(PYTHON) -m $(PYTEST) $(TEST_AA) -v

lint:
	@$(PYTHON) -m pip install -q flake8
	@$(PYTHON) -m flake8

# Create virtual enviorment def .venv folder
venv:
	@$(PYTHON) -m venv .venv
	@echo "Virtualk enviorment created in .venv/"