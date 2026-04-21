PYTHON ?= python

.PHONY: install test check analysis benchmark train

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-base.txt pytest
	$(PYTHON) -m pip install -r requirements-jax.txt

test:
	$(PYTHON) -m pytest -q code/tests

check:
	$(PYTHON) -m compileall code/benchmarks code/code_experiments code/src code/src_jax code/tests

analysis:
	$(PYTHON) code/code_experiments/plot_jax_vs_pytorch_comparison.py

benchmark:
	$(PYTHON) code/benchmarks/aggregate_results.py

train:
	$(PYTHON) code/benchmarks/pytorch/train_fulltrainer_benchmark.py
