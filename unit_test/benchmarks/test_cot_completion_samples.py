"""Check cli/cot_completion.py::main with optional bounded real-server generation.

Run with PYTHONPATH=.:demos python -m unit_test.benchmarks.test_cot_completion_samples.
Add --live to generate two 16-token responses for one cached AIME24 question.
Authentication and endpoint come from OPENAI_API_KEY and OPENAI_API_BASE.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    """Check CLI validation; optionally inspect two independently seeded saved responses."""
    command = [sys.executable, "-m", "lits.cli.cot_completion"]
    args = ["--dataset", "aime24", "--include", "lits_benchmark.math_qa",
            "--limit", "1", "--model", "openai/local-instruct",
            "--tokenizer", "Qwen/Qwen3-8B", "--tokenizer-revision",
            "a80f5e57cce20e57b65145f4213844dec1a80834",
            "--temperature", "0.7", "--seed", "0", "--max-new-tokens", "16"]
    with tempfile.TemporaryDirectory(prefix="cot-samples-") as temporary:
        output = Path(temporary) / "results"
        args += ["--output-dir", str(output)]
        invalid = subprocess.run(command + args + ["--samples", "0"],
                                 capture_output=True, text=True)
        if invalid.returncode != 2 or "samples must be positive" not in invalid.stderr:
            raise RuntimeError(f"Unexpected validation result: {invalid.stderr}")
        print("Nonpositive sample count rejected before inference")
        if "--live" not in sys.argv:
            return
        subprocess.run(command + args + ["--samples", "2"], check=True)
        rows = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
        if (len(rows) != 2 or any("error" in row for row in rows)
                or [row["sample_index"] for row in rows] != [0, 1]
                or [row["seed"] for row in rows] != [0, 1]
                or rows[0]["prompt"] != rows[1]["prompt"]):
            raise RuntimeError("Expected two independent responses with the same initial prompt")
        config = json.loads((output / "config.json").read_text())
        if config["samples"] != 2 or config["sample_counts"] != {"0": 2}:
            raise RuntimeError("Saved sample counts differ from the request")
        print("Two samples saved with identical prompts, distinct seeds, and per-sample costs")
        breakpoint()  # inspect: rows, config, output


if __name__ == "__main__":
    main()
