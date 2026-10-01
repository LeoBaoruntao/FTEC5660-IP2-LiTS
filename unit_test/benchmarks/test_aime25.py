"""Check math_qa.py::load_aime25 through the public registry, without LLM inference.

Run with PYTHONPATH=.:demos PYTHONBREAKPOINT=0 python -m unit_test.benchmarks.test_aime25.
The first run downloads the real dataset; subsequent runs can use HF_HUB_OFFLINE=1.
"""

import datasets

from lits.registry import import_custom_modules
from lits.benchmarks.registry import load_dataset


def main():
    """Verify full coverage, original references, IDs and source ordering."""
    import_custom_modules(["lits_benchmark.math_qa"])
    rows = load_dataset("aime25")
    source = datasets.load_dataset("math-ai/aime25", split="test",
                                   revision="563bb8404243c5f09de6ec262f2db674fe5bce9b")
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise ValueError("Expected 30 distinct AIME25 questions")
    for row, original in zip(rows, source):
        if row != dict(question=original["problem"], answer=original["answer"], id=original["id"]):
            raise ValueError("Question, reference, ID or ordering changed")
        if not row["question"] or not row["answer"].isdigit() or not 0 <= int(row["answer"]) <= 999:
            raise ValueError("Invalid AIME question or integer reference")
    print("Passed: 30 registered AIME25 questions; source order, IDs and references preserved")
    breakpoint()  # inspect: rows[0], rows[-1]


if __name__ == "__main__":
    main()
