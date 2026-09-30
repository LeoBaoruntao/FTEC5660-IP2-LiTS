"""Generate one complete non-thinking CoT response through raw completion.

Run ``python -m lits.cli.cot_completion --dataset math500
--include lits_benchmark.math_qa --limit 100 --model openai/local-instruct
--tokenizer Qwen/Qwen3-8B --tokenizer-revision COMMIT --output-dir NEW_DIRECTORY``.
The cached tokenizer must match the server. Use the registered continuation
system instruction and a separate user query; never include reference answers.
Each example uses seed + its original dataset index.

benchmarks.cot.run_cot owns dataset selection, progress, costs and immediate saving.
Responses retain raw text, rendered prompt, seed, finish reason and token counts.
Config retains resolved settings, tokenizer revision and server model metadata.
Evaluate with ``python -m lits.cli.eval_cot --result-dir DIRECTORY``; no separate
completion evaluator is needed. Chat and multi-sample generation stay in cli.cot.

MATH-100 commands, from the repository root with OPENAI_API_KEY and
OPENAI_API_BASE set to the serving endpoint. The tokenizer must already be cached;
the registered numeric-answer dataset uses its first 100 examples, indices 0-99::

    export PYTHONPATH=.:demos
    OUT="results/math100-cot-$(date +%Y%m%d-%H%M%S)"
    python -m lits.cli.cot_completion \
      --dataset math500 --include lits_benchmark.math_qa --offset 0 --limit 100 \
      --model openai/local-instruct --base-url "$OPENAI_API_BASE" \
      --tokenizer Qwen/Qwen3-8B \
      --tokenizer-revision a80f5e57cce20e57b65145f4213844dec1a80834 \
      --temperature 0.6 --top-p 0.8 --top-k 20 --min-p 0 --seed 0 \
      --max-new-tokens 8192 --output-dir "$OUT"
    python -m lits.cli.eval_cot --result-dir "$OUT"

The second command evaluates saved outputs without rerunning generation. Both
commands show progress. Keep OUT unchanged between them; generation requires a
fresh directory, while each evaluation creates its own timestamped subdirectory.
"""

import argparse

from transformers import AutoTokenizer

from lits.benchmarks.cot import run_cot
from lits.cli.args import parse_dataset_kwargs
from lits.components.utils import create_role
from lits.lm.openai_completion import OpenAICompletionModel
from lits.prompts.registry import PromptRegistry
from lits.registry import import_custom_modules


def run(args):
    """Configure raw generation and delegate execution/saving to run_cot.

    Args comes from main(). Authentication and endpoint also accept the standard
    OPENAI_API_KEY and OPENAI_API_BASE environment variables. No search or scoring.
    """
    import_custom_modules(args.import_modules)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, local_files_only=True,
    )
    system_prompt = PromptRegistry.get(
        "policy", "continuation", task_name=args.dataset, task_type="language_grounded",
    )
    model = OpenAICompletionModel(args.model.removeprefix("openai/"), base_url=args.base_url)
    config = {**vars(args), "samples": 1, "thinking": False,
              "system_prompt": system_prompt, "model_id": model.model_name,
              "base_url": str(model.client.base_url),
              "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or args.tokenizer_revision,
              "model_metadata": model.client.models.list().model_dump()}

    def generate(question, index, repeat):
        """Render once and return untrimmed completion text with generation evidence."""
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        seed = args.seed + index
        response = model(prompt, role=create_role("policy", index), seed=seed,
                         temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                         top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)
        return dict(raw_response=response.text, prompt=prompt, seed=seed,
                    finish_reason=response.finish_reason, input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens)

    run_cot(config, model, generate, dataset_kwargs=parse_dataset_kwargs(args))


def main():
    """Parse raw single-pass settings; chat-only sampling options are not exposed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--include", dest="import_modules", nargs="+")
    parser.add_argument("--dataset-arg", dest="dataset_args", nargs="+", action="extend")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True, help="Pinned tokenizer commit.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0:
        parser.error("max-new-tokens must be positive and temperature nonnegative")
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        parser.error("offset and limit must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
