# Custom Evaluators

Register dataset-specific evaluation functions so `lits-eval` uses your comparison logic instead of the default numeric matcher.

## Contents

- [Why](#why)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Example: DBBench](#example-dbbench)
- [In-Memory Evaluation](#in-memory-evaluation)
- [API](#api)
- [Dataset Loader Convention](#dataset-loader-convention)
- [Evaluator Signature Contract](#evaluator-signature-contract)

## Why

The default `lits-eval` accuracy check uses `eval_output(type="number_exact")`, which works for math QA but not for benchmarks that need float tolerance, set comparison, or special value normalization (e.g., DBBench, KG F1).

## Usage

```python
from lits.benchmarks.registry import register_evaluator

@register_evaluator("my_dataset")
def evaluate_my_dataset(predicted, ground_truth) -> bool:
    """Return True if predicted matches ground_truth."""
    return predicted.strip().lower() == ground_truth.strip().lower()
```

`lits-eval` automatically picks up the registered evaluator when evaluating results for `my_dataset`. No CLI flags needed — just ensure the module is imported (via `--include`).

## How It Works

After output selection, correctness checking follows this order:

1. `env_grounded`: goal check; other correctness evaluators are ignored.
2. Registered evaluator: use its bool/float result, with configured LLM fallback on
   non-full matches or exceptions. Only a float score of `1.0` counts as correct.
3. Without a registered evaluator: use the configured LLM judge; otherwise use
   `eval_output(type="number_exact")`, with exact-string fallback on numeric parsing errors.

CLI defaults are unchanged: tool-use tasks enable `--llm-eval binary`; use `f1` for
continuous judging or `none` to disable it. F1 fallback retains the larger registered/LLM
score; binary fallback changes correctness without replacing the registered score.

## Example: DBBench

DBBench needs float tolerance (±0.01), set comparison for multi-value answers, and special value normalization (`None` → `"0"`, `NaN` → `"0"`):

```python
from lits.benchmarks.registry import register_evaluator

@register_evaluator("dbbench")
def evaluate_dbbench(predicted_answer, ground_truth) -> bool:
    # Float tolerance for numeric answers
    # Set comparison for multi-value results
    # Special value normalization (None, NaN, Inf → "0")
    ...
```

Run evaluation:

```bash
lits-eval --result_dir results/dbbench_rest/run_0 \
    --include lits_benchmark.dbbench
```

## In-Memory Evaluation

Use the same tree evaluation without saving/reloading files. Given a completed search
`result`, its `query`, and the dataset `example`:

```python
import lits_benchmark.dbbench  # register the benchmark evaluator
from lits.benchmarks.registry import get_evaluator
from lits.eval.search import evaluate_search_result

outcome = evaluate_search_result(
    result.terminal_nodes_collected,
    query,
    example["answer"],
    task_type="tool_use",
    custom_evaluator=get_evaluator("dbbench"),
)
```

The caller supplies terminal nodes, task type, query/reference, and configured callbacks.
Benchmark imports, dataset/resource setup, and model creation remain outside this function.
The CLI calls it after loading saved tree terminals; chain/pass-at-N loading stays in the CLI.

**Output selection chooses what to score; correctness checking compares that output with
the reference.** The reference never chooses the terminal or vote winner.

| `task_type` | Output selection (unchanged defaults) | Caller setup |
|---|---|---|
| `language_grounded` | Vote over extracted terminal answers | Supply `retrieve_answer(state, query)`, e.g. `get_fn_retrieve_answer(model)` from `lits.components.utils` |
| `tool_use` | Default `terminal_selection="q-mean"`; also `q-max` or `first` | Supply the registered checker; answer retrieval from the selected node is built in |
| `env_grounded` | Maximum backed-up terminal return | Supply `goal_check(query, final_env_state)` returning `(goal_reached, progress)` |

`q-mean`/`q-max` rank `cum_rewards`, not `fast_reward`. Missing/empty returns rank as
negative infinity; ties retain input order, including when all returns are absent.

Unlike the CLI, direct calls default to `llm_evaluator=None`: no LLM judge is created.
To enable judging, supply a configured `lits.eval.general_eval.GeneralEvaluator` and
matching `llm_eval_mode`: `"binary"` with a `correct` yes/no perspective, or `"f1"`
with a float `score` perspective. Answer-extraction callbacks may independently call an LLM.

| Returned field | Meaning |
|---|---|
| `prediction` | Selected/voted answer; `"correct"` or `"incorrect"` for environment tasks |
| `correct` | Boolean correctness of that output, not whether any terminal is correct |
| `score` | Continuous checker score or `None`; not a search reward or `float(correct)` |
| `environment_progress` | Goal-check progress for environment tasks; otherwise `None` |

Empty terminals yield `"incorrect"` and progress `0.0` for environment tasks;
other tasks pass an empty prediction through their configured checker. This preserves
benchmark-specific empty-answer behavior. Required callbacks must still be supplied.
Full parameter details: [`evaluate_search_result`](../../lits/eval/search.py).

## API

```python
from lits.benchmarks.registry import register_evaluator, get_evaluator, has_evaluator

# Register
@register_evaluator("name")
def my_eval(predicted, ground_truth) -> bool: ...

# Query
has_evaluator("name")       # True / False
get_evaluator("name")       # callable or None
```

## Dataset Loader Convention

For `lits-eval` compatibility, dataset loaders should include:

- `"question"` key — used by `lits-search` CLI to extract the query
- `"answer"` key — used by `lits-eval` to look up ground truth

These can be aliases of domain-specific fields (e.g., `"question"` = `"description"`, `"answer"` = `"label"` for DBBench).

## Evaluator Signature Contract

```python
@register_evaluator("my_dataset")
def my_evaluator(predicted: str, ground_truth: Any) -> Union[bool, float]:
    ...
```

- `predicted` — always a `str` (the agent's committed answer from the checkpoint).
- `ground_truth` — **the raw value from `example["answer"]`** in your dataset loader. It is passed directly without any type conversion. If your dataset loader returns a dict (e.g., `{"names": [...], "ids": [...]}`), your evaluator receives that dict. If it returns a string, your evaluator receives a string.
- Return `bool` (correct/incorrect) or `float` (e.g., F1 score where 1.0 = exact match).

This means the evaluator and dataset loader must agree on the `ground_truth` type. For example:

| Dataset | `example["answer"]` type | Evaluator handles |
|---------|--------------------------|-------------------|
| KGQA | `dict` (`{"names": [...], "ids": [...]}`) | Extracts IDs, does set comparison after splitting predicted by comma |
| DBBench | `str` or `list[str]` | Float tolerance, set comparison |
| Math QA | `str` (numeric) | Default numeric comparison |
