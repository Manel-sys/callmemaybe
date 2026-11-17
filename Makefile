MAIN = src
PY = python3
ARGS ?=

# Make uv findable right after the curl installer puts it in ~/.local/bin
export PATH := $(HOME)/.local/bin:$(PATH)

# ---- Heavy storage: only redirected when sgoinfre exists ----
SGOINFRE ?= /sgoinfre/$(USER)

ifneq ($(wildcard $(SGOINFRE)),)
    HEAVY_DIR := $(SGOINFRE)/call_me_maybe
    VENV_DIR := $(HEAVY_DIR)/venv
    export UV_CACHE_DIR := $(HEAVY_DIR)/uv_cache
    export UV_PYTHON_INSTALL_DIR := $(HEAVY_DIR)/python
    export HF_HOME := $(HEAVY_DIR)/hf_cache
endif

EXCLUDES = .venv,llm_sdk
MYPY_EXCLUDE = '\.venv|llm_sdk'

install:
	@command -v uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
ifdef HEAVY_DIR
	@echo "sgoinfre detected, heavy files go to: $(HEAVY_DIR)"
	@mkdir -p $(HEAVY_DIR) && chmod 700 $(HEAVY_DIR)
	@if [ -d .venv ] && [ ! -L .venv ]; then \
		echo "Removing local .venv (it would waste home quota)"; \
		rm -rf .venv; \
	fi
	UV_PROJECT_ENVIRONMENT=$(VENV_DIR) uv sync
	@ln -sfn $(VENV_DIR) .venv
	@echo ".venv -> $(VENV_DIR)"
else
	@echo "sgoinfre not found, using default locations (.venv, ~/.cache)"
	uv sync
endif

run:
	uv run $(PY) -m $(MAIN) $(ARGS)

debug:
	uv run $(PY) -m pdb -m $(MAIN) $(ARGS)

clean:
	find . -path ./.venv -prune -o -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -name "*.pyc" -type f -exec rm -f {} +
	rm -rf .mypy_cache .pytest_cache

fclean: clean
ifdef HEAVY_DIR
	rm -rf $(HEAVY_DIR)
endif
	rm -rf .venv

lint:
	uv run flake8 --exclude=$(EXCLUDES) .
	uv run mypy . --exclude $(MYPY_EXCLUDE) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 --exclude=$(EXCLUDES) .
	uv run mypy . --exclude $(MYPY_EXCLUDE) --strict

info:
	@echo "HEAVY_DIR    = $(or $(HEAVY_DIR),(not set - no sgoinfre))"
	@echo ".venv        = $$(readlink .venv 2>/dev/null || echo 'local folder or missing')"
	@echo "UV_CACHE_DIR = $(or $(UV_CACHE_DIR),~/.cache/uv (default))"
	@echo "HF_HOME      = $(or $(HF_HOME),~/.cache/huggingface (default))"

.PHONY: install run debug clean fclean lint lint-strict info