"""Offline checks for the registered math system prompt and fixed MATH-100 cohort.

Run from lits_llm with the cached dataset and tokenizer::

    PYTHONPATH=.:demos HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONBREAKPOINT=0 \
        python -m unit_test.benchmarks.test_math_prompt \
        --cohort /path/to/saved/cohort.json \
        --reference /path/to/prepare_data.py

Compare registered dataset indices 0-99 with the original cohort and load only
the reference apply_chat_template function, avoiding its unrelated dependencies.
Exercise continuation.py::ContinuationPolicy.from_config/_build_messages with a
real client and cached tokenizer, without making any model requests.
"""

import argparse
import ast
import json
from pathlib import Path

from transformers import AutoTokenizer

from lits.benchmarks.registry import load_dataset
from lits.benchmarks.runner import _slice_dataset
from lits.components.policy.continuation import ContinuationPolicy
from lits.lm.openai_completion import OpenAICompletionModel
from lits.prompts.registry import PromptRegistry
import lits_benchmark.math_qa  # noqa: F401 -- registers the dataset
from unit_test.lm.test_raw_completion_server import TOKENIZER_REVISION


def check_math_prompt(cohort: Path, reference: Path):
    """Verify cohort identity, reference rendering and inherited prompt overrides."""
    saved = json.loads(cohort.read_text())
    rows = load_dataset("math500")
    indices = _slice_dataset(list(range(len(rows))), 0, 100)
    if indices != list(range(100)) or [row["query_idx"] for row in saved] != indices:
        raise RuntimeError("Expected the saved cohort to contain indices 0-99 in source order")
    for index in indices:
        for field in ("unique_id", "question", "answer"):
            if rows[index][field] != saved[index]["example"][field]:
                raise RuntimeError(f"Cohort mismatch: example {index}, field {field}")

    functions = [node for node in ast.parse(reference.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "apply_chat_template"]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(reference), "exec"), namespace)
    reference_prompt = namespace["apply_chat_template"]
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    model = OpenAICompletionModel("local-instruct", api_key="no-key",
                                  base_url="http://127.0.0.1:8080/v1")
    model.tokenizer = tokenizer
    try:
        policy = ContinuationPolicy.from_config(
            model, {"n_actions": 1}, {}, dataset_name="math500", task_name="math_qa",
        )
        system_prompt = PromptRegistry.get("policy", "continuation", task_type="language_grounded")
        if not isinstance(system_prompt, str) or policy.task_prompt_spec != system_prompt:
            raise RuntimeError("Policy did not load the registered system instruction")
        for index in indices:
            query = rows[index]["question"]
            expected = reference_prompt(rows[index], tokenizer, "math", no_thinking=True)
            messages = policy._build_messages(query)
            if messages != [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": query}]:
                raise RuntimeError("System instruction and query are not separate messages")
            actual = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            if actual != expected:
                raise RuntimeError(f"Prompt differs from the reference for example {index}")
        default = ContinuationPolicy.from_config(model, {"n_actions": 1}, {})
        if default._build_messages(query) != policy._build_messages(query):
            raise RuntimeError("General continuation prompt depends on a dataset registration")
        override = PromptRegistry.get("policy", "concat", task_type="language_grounded")
        explicit = ContinuationPolicy.from_config(
            model, {"n_actions": 1}, {}, dataset_name="math500", task_prompt_spec=override,
        )
        if not override or explicit._build_system_prompt() != override:
            raise RuntimeError("Explicit system instruction did not override the registry")
        omitted = ContinuationPolicy(model, tokenizer, n_actions=1,
                                     task_name="math500", task_prompt_spec="")
        if omitted._build_messages(query) != [{"role": "user", "content": query}]:
            raise RuntimeError("An empty instruction did not suppress the registered prompt")
        breakpoint()  # inspect: indices, rows[0], saved[0], actual, expected, policy
    finally:
        model.client.close()
    print("Passed: 100 cohort identities/questions/answers, 100 reference prompts, "
          "system/user roles, general default and explicit overrides; no inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    check_math_prompt(args.cohort, args.reference)
