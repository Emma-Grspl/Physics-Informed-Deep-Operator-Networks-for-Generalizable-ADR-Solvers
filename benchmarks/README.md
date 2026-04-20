# Benchmark Layout

This benchmark stack is the shared evaluation layer used by the PyTorch versus JAX comparison workflows.

Goals:
- compare PyTorch and JAX under the same short-horizon protocol
- measure training speed, convergence, CN-relative L2, inference speed, and time-jump speed
- keep outputs under `outputs/benchmarks/`

Main components:

- `common/`: shared configuration, I/O, case generation, and evaluation helpers
- `pytorch/`: PyTorch benchmark entry points
- `jax/`: JAX benchmark entry points
- `aggregate_results.py`: post-processing utility for benchmark outputs

Suggested workflow:
1. run `benchmarks/pytorch/train_short_benchmark.py`
2. run `benchmarks/jax/train_short_benchmark.py`
3. run the corresponding `eval_benchmark.py`
4. run the corresponding `inference_benchmark.py`
5. aggregate with `benchmarks/aggregate_results.py`

For human-facing experiment definitions, use [experiments/README.md](../experiments/README.md). The `benchmarks/` tree contains the execution helpers, not the public protocol registry.
