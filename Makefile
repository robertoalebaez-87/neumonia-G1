ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
else
SHELL := /bin/bash
.SHELLFLAGS := -ec
endif

.DEFAULT_GOAL := help

.PHONY: help install dev run lint format format-check typecheck test test-cov check pre-commit build clean

help: ## Lista los objetivos disponibles
	@uv run python tools/make_help.py

install: ## Instala/actualiza el entorno con uv (incluye el extra ml)
	uv sync --extra ml

dev: install ## Instala el entorno y los hooks de pre-commit
	uv run pre-commit install

run: ## Inicia la interfaz web en http://127.0.0.1:8000
	uv run python detector_neumonia.py

lint: ## Ejecuta Ruff (lint)
	uv run ruff check .

format: ## Formatea el código con Ruff
	uv run ruff format .

format-check: ## Verifica el formato sin modificar
	uv run ruff format --check .

typecheck: ## Comprueba tipos con ty
	uv run ty check

test: ## Ejecuta las pruebas unitarias
	uv run pytest

test-cov: ## Ejecuta las pruebas con cobertura
	uv run pytest --cov --cov-report=term-missing

check: lint format-check typecheck test ## Calidad completa: lint + formato + tipos + pruebas

pre-commit: ## Ejecuta todos los hooks de pre-commit
	uv run pre-commit run --all-files

build: ## Construye sdist y wheel
	uv build

clean: ## Limpia cachés y artefactos
	@uv run python tools/clean.py
