"""Verify registered calculus and chemistry datasets without model inference.

Run with PYTHONPATH=.:demos python -m unit_test.benchmarks.test_scibench_calculus.
The check downloads the public source and verifies a local-copy round trip.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lits.benchmarks.registry import load_dataset, infer_task_type, get_evaluator
import lits_benchmark.math_qa  # noqa: F401 -- registers the dataset


def check_scibench_dataset(name, expected_count):
    """Check math_qa.py::_load_scibench through both public dataset registrations."""
    rows = load_dataset(name)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} examples for {name}, found {len(rows)}")
    if infer_task_type(name) != "language_grounded" or get_evaluator(name) is not None:
        raise RuntimeError("SciBench must use language reasoning without the MATH checker")
    if any(not isinstance(row["question"], str) or not row["question"].strip()
           or not isinstance(row["answer"], str) for row in rows):
        raise RuntimeError("Invalid question/answer fields")
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.json"
        path.write_text("\n".join(json.dumps({"content": row["question"], "answer": row["answer"]})
                                  for row in rows), encoding="utf-8")
        reloaded = load_dataset(name, path=str(path))
        if rows != reloaded:
            raise RuntimeError("Local loading changed content or order")
        if name == "scibench_chemmc":
            if rows[3]["answer"] != "1/3":
                raise RuntimeError("Alternating-series reference was not corrected")
            source = [{"content": rows[3]["question"], "answer": 1.0},
                      {"content": "A different series", "answer": 1.0}]
            path.write_text("\n".join(map(json.dumps, reversed(source))), encoding="utf-8")
            corrected = load_dataset(name, path=str(path))
            if [row["answer"] for row in corrected] != ["1.0", "1/3"]:
                raise RuntimeError("Reference correction depends on position or changes unrelated answers")
    print(f"PASS: {name}, {len(rows)} examples; string answers; local/remote loading agree")
    breakpoint()  # inspect: rows; source order and unit instructions are retained


if __name__ == "__main__":
    check_scibench_dataset("scibench_calculus", 42)
    check_scibench_dataset("scibench_chemmc", 39)
