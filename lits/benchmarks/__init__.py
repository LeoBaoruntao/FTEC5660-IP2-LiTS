"""Benchmark registration and sequential execution for LiTS.

Provides dataset registration, task type inference, and caller-controlled execution.
"""

from lits.benchmarks.registry import (
    BenchmarkRegistry,
    register_dataset,
    load_dataset,
    infer_task_type,
    register_evaluator,
    get_evaluator,
    has_evaluator,
)
from lits.benchmarks.runner import run_benchmark
from lits.benchmarks.search import run_example

__all__ = [
    'BenchmarkRegistry',
    'register_dataset',
    'load_dataset',
    'infer_task_type',
    'register_evaluator',
    'get_evaluator',
    'has_evaluator',
    'run_benchmark',
    'run_example',
]
