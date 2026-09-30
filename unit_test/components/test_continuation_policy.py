"""Bounded real-server checks for continuation.py::ContinuationPolicy.get_actions.

Run from lits_llm with OPENAI_API_BASE and OPENAI_API_KEY configured::

    PYTHONBREAKPOINT=0 python -m unit_test.components.test_continuation_policy \
        --output-dir /path/to/new/evidence

Five requests use at most 96 output tokens. Capture wire prompts, check retained
newline boundaries and seeds, and round-trip qa.py::ContinuationStep through
base.py::TrajectoryState.from_dict. Exercise concat.py::ConcatTransition.step
and is_terminal without another LLM call. No benchmark run or model mocks.
"""

import argparse
import json
import os
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from lits.components.policy.continuation import ContinuationPolicy
from lits.components.transition.concat import ConcatTransition
from lits.lm.openai_completion import OpenAICompletionModel
from lits.structures import TrajectoryState
from unit_test.lm.test_raw_completion_server import TOKENIZER_REVISION, save


def check_policy(base_url: str, output_dir: Path):
    """Verify actual requests and serializable chunks; retain evidence on failure."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    captured = []

    def capture(response):
        """Save real traffic without authentication headers."""
        response.read()
        captured.append({"request": json.loads(response.request.content),
                         "response": response.json(), "url": str(response.request.url)})
        save(output_dir / f"http_{len(captured)}.json", captured[-1])

    with httpx.Client(event_hooks={"response": [capture]}, timeout=120) as client:
        model = OpenAICompletionModel("local-instruct", base_url=base_url)
        model.client = model.client.with_options(http_client=client, timeout=120, max_retries=0)
        policy = ContinuationPolicy(model, tokenizer, n_actions=2, seed=41, max_new_tokens=16)
        policy.max_steps = 0  # Depth must not reduce candidate count or temperature.
        query = "List integers 1 through 100, one integer per line, with no other text."
        initial_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": policy.task_prompt_spec},
             {"role": "user", "content": query}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        roots = policy.get_actions([], query=query, query_idx=0)
        transition = ConcatTransition(model, terminate_constraints=["step_signal"])
        state, _ = transition.step([], roots[0], query, query_idx=0)
        if transition.is_terminal(state, query, query_idx=0):
            raise RuntimeError("Expected a token-limited, nonterminal first chunk")
        children = policy.get_actions(state, query=query, query_idx=0)
        policy.max_new_tokens = 32
        final = policy.get_actions([], query="Reply with OK only.", n_actions=1, query_idx=1)
        steps = roots + children + final
        save(output_dir / "steps.json", [step.to_dict() for step in steps])

        for index, (step, exchange) in enumerate(zip(steps, captured)):
            request, response = exchange["request"], exchange["response"]
            choice = response["choices"][0]
            text = choice["text"]
            # The source keeps the last newline but charges the full generation.
            boundary = max(text.rfind("\n\n"), text.rfind("\n"))
            retained = text[:boundary + 1] if choice["finish_reason"] == "length" and boundary > 0 else text
            if (step.action != retained or step.raw_text != text
                    or step.output_tokens != response["usage"]["completion_tokens"]
                    or step.input_tokens != response["usage"]["prompt_tokens"]
                    or step.finish_reason != choice["finish_reason"]):
                raise RuntimeError("Chunk text or metadata differs from server evidence")
            if (step.seed != 41 + index or request["seed"] != step.seed
                    or request["n"] != 1 or request["temperature"] != 0.7
                    or request["top_p"] != 0.8 or request["top_k"] != 20
                    or request["min_p"] != 0.0 or "messages" in request):
                raise RuntimeError("Sampling parameters changed")
            if index < 4:
                expected = initial_prompt + (roots[0].action if index >= 2 else "")
                if request["prompt"] != expected or step.finish_reason != "length":
                    raise RuntimeError("Continuation prefix or length cutoff differs")
        if len(roots) != 2 or len(children) != 2 or len(captured) != 5:
            raise RuntimeError("Depth changed the requested candidate count")
        final_state, _ = transition.step([], final[0], "Reply with OK only.", query_idx=1)
        if not transition.is_terminal(final_state, "Reply with OK only.", query_idx=1):
            raise RuntimeError("Natural completion did not set the terminal signal")
        restored = TrajectoryState.from_dict(json.loads(json.dumps(TrajectoryState(steps).to_dict())))
        if restored != steps:
            raise RuntimeError("Checkpoint round-trip lost chunk metadata")
        try:
            policy.get_actions(final_state, query="Reply with OK only.")
        except ValueError:
            pass
        else:
            raise RuntimeError("Expanded a completed trajectory")
        breakpoint()  # inspect: roots, children, final, captured, restored
    print(f"Passed continuation, seeds, cutoffs, transition and serialization checks: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE"), required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Supply --base-url or OPENAI_API_BASE")
    check_policy(args.base_url, args.output_dir)
