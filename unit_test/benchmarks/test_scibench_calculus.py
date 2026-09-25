"""Verify the registered calculus dataset without model inference.

Run with PYTHONPATH=.:demos python -m unit_test.benchmarks.test_scibench_calculus.
The check downloads the public source and verifies a local-copy round trip.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lits.benchmarks.registry import load_dataset
import lits_benchmark.math_qa  # noqa: F401 -- registers the dataset


def check_calculus_dataset():
    """Check math_qa.py::load_scibench_calculus through the dataset registry."""
    rows = load_dataset("scibench_calculus")
    if len(rows) != 42:
        raise RuntimeError(f"Expected the agreed 42-example cohort, found {len(rows)}")
    if any(not isinstance(row["question"], str) or not row["question"].strip()
           or not isinstance(row["answer"], str) for row in rows):
        raise RuntimeError("Invalid question/answer fields")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "calculus.json"
        path.write_text("\n".join(json.dumps({"content": row["question"], "answer": row["answer"]})
                                  for row in rows), encoding="utf-8")
        reloaded = load_dataset("scibench_calculus", path=str(path))
        if rows != reloaded:
            raise RuntimeError("Local loading changed content or order")
    print(f"PASS: {len(rows)} examples; string answers; local/remote loading agree")
    breakpoint()  # inspect: rows; source order and unit instructions are retained


if __name__ == "__main__":
    check_calculus_dataset()
