.PHONY: install dev lint type test smoke train eval docker clean

PY ?= python
CONFIG ?= configs/default.yaml

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest --cov=tft --cov-report=term-missing

smoke:
	WANDB_MODE=disabled tft-train --config configs/smoke.yaml

train:
	tft-train --config $(CONFIG)

eval:
	tft-eval --config $(CONFIG) --checkpoint checkpoints/best.pt

docker:
	docker build -t tft-transit:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache mlruns wandb
