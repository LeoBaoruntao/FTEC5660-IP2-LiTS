"""Bounded checks for cot.py::run and cot_completion.py::run, without evaluation.

Run with PYTHONPATH=.:demos, OPENAI_API_KEY and PYTHONBREAKPOINT=0::

    python -m unit_test.cli.test_cot_generation --base-url SERVER/v1 \
        --output-dir /path/to/new/evidence

Generate one raw sample and two independent chat samples for dataset index 14,
at most 48 output tokens in total. Use the cached dataset/tokenizer; retain the
production output files for inspection. No mock models or response substitutions.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from lits.cli.cot import PROMPT, run
from lits.cli.cot_completion import run as run_completion
from lits.constants import DEFAULT_REASONING_TEMPERATURE
from lits.prompts.registry import PromptRegistry
from unit_test.lm.test_raw_completion_server import TOKENIZER_REVISION


def check_generation(base_url: str, output_dir: Path):
    """Check original IDs, sampling settings, raw evidence and chat compatibility."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    for raw in (True, False):
        directory = output_dir / ("raw" if raw else "chat")
        args = argparse.Namespace(
            dataset="math500", import_modules=["lits_benchmark.math_qa"], dataset_args=None,
            offset=14, limit=1, model="openai/local-instruct", base_url=base_url,
            temperature=0.6 if raw else DEFAULT_REASONING_TEMPERATURE,
            max_new_tokens=16, output_dir=str(directory),
        )
        samples = 1 if raw else 2
        if raw:
            args.tokenizer, args.tokenizer_revision = "Qwen/Qwen3-8B", TOKENIZER_REVISION
            args.seed, args.top_p, args.top_k, args.min_p = 7, 0.8, 20, 0.0
            run_completion(args)
        else:
            args.samples, args.sample_counts, args.thinking = samples, None, False
            run(args)
        config = json.loads((directory / "config.json").read_text())
        cohort = json.loads((directory / "cohort.json").read_text())
        records = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
        if (config["sample_counts"] != {"14": samples} or len(cohort) != 1
                or cohort[0]["query_idx"] != 14
                or [(row["query_idx"], row["sample_index"]) for row in records]
                != [(14, sample) for sample in range(samples)]):
            raise RuntimeError("Original example identity or sample counts changed")
        if any("error" in row or not row["raw_response"]
               or row["generation_costs"]["num_calls"] != 1
               or row["generation_seconds"] <= 0 for row in records):
            raise RuntimeError(f"Inspect {directory}: generation or per-call accounting failed")
        if raw:
            system = PromptRegistry.get("policy", "continuation", task_type="language_grounded")
            expected = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": cohort[0]["example"]["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            record = records[0]
            if (record["prompt"] != expected or record["seed"] != 21
                    or record["finish_reason"] not in ("stop", "length")
                    or not 0 < record["output_tokens"] <= 16 or record["input_tokens"] <= 0
                    or config["tokenizer_revision"] != TOKENIZER_REVISION
                    or config["system_prompt"] != system or not config["model_metadata"]):
                raise RuntimeError("Raw prompt, seed, stopping or model metadata was lost")
            for key in ("seed", "temperature", "top_p", "top_k", "min_p", "max_new_tokens"):
                if config[key] != getattr(args, key):
                    raise RuntimeError(f"Saved setting changed: {key}")
        elif config["prompt"] != PROMPT or any("finish_reason" in row for row in records):
            raise RuntimeError("Existing chat generation format changed")
        breakpoint()  # inspect: args, config, cohort, records
    print(f"Passed raw single-pass and chat multi-sample generation: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check_generation(args.base_url, args.output_dir)
