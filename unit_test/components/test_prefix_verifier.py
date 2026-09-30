"""Check prefix_verifier.py::PrefixVerifier against saved real server responses.

Run with --reference pointing to the supplied llm_as_verifier.py and --saved-dir
pointing to the successful raw-completion server probe. --live additionally tests
factory.py::create_components with one generation and two verifications (at most
3088 output tokens). Requires OPENAI_API_KEY and a server-matching cached tokenizer.
Use PYTHONBREAKPOINT=0 for unattended execution and a fresh --output-dir.

Only the reference prompt/scoring functions are loaded, not its server imports.
Saved token evidence is adapted to their vLLM-shaped input; no model is mocked.
"""

import argparse
import ast
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
from transformers import AutoTokenizer

from lits.components.factory import create_components
from lits.components.policy.continuation import ContinuationPolicy
from lits.components.reward.prefix_verifier import PrefixVerifier
from lits.config import ExperimentConfig
from lits.lm.openai_completion import OpenAICompletionModel
from lits.structures import ThoughtStep
from unit_test.lm.test_raw_completion_server import TOKENIZER_REVISION, save


def load_reference(path: Path):
    """Load only the source's prompt builder and scorer for direct parity checks."""
    parsed = ast.parse(path.read_text())
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("build_llm_as_verifier_prompt", "verification_score")]
    namespace = {"np": math, "RequestOutput": object}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_llm_as_verifier_prompt"], namespace["verification_score"]


def compare_score(logprobs, reference_score):
    """Compare the exact same saved token alternatives, including their order."""
    positions = [{i: SimpleNamespace(decoded_token=token["token"], logprob=token["logprob"])
                  for i, token in enumerate(position["top_logprobs"])}
                 for position in logprobs["content"]]
    expected, expected_token = reference_score(SimpleNamespace(
        outputs=[SimpleNamespace(logprobs=positions)]))
    score, token = PrefixVerifier.verification_score(logprobs)
    if score != expected or token["token"] != expected_token:
        raise RuntimeError("Token score differs from the reference")
    return score


def check_verifier(args):
    """Replay scores offline, optionally check real factory-built components."""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build_prompt, reference_score = load_reference(args.reference)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    query = "What is the sum of 15 and 27?"
    scores = {}
    for label in ("Yes", "No"):
        saved = json.loads((args.saved_dir / f"verification_{label}.json").read_text())
        response = json.loads(saved["response_text"])
        scores[label] = compare_score(response["choices"][0]["logprobs"], reference_score)
    save(args.output_dir / "saved_scores.json", scores)
    if not args.live:
        print(f"Saved-response parity passed: {scores}")
        return

    captured = []

    def capture(response):
        """Retain real wire data without authentication headers."""
        response.read()
        captured.append({"request": json.loads(response.request.content), "response": response.json()})
        save(args.output_dir / f"http_{len(captured)}.json", captured[-1])

    with httpx.Client(event_hooks={"response": [capture]}, timeout=120) as client:
        model = OpenAICompletionModel("local-instruct", base_url=args.base_url)
        model.client = model.client.with_options(http_client=client, timeout=120, max_retries=0)
        model.tokenizer = tokenizer
        config = ExperimentConfig(
            dataset="math500", policy="continuation", transition="concat", reward="prefix_verifier",
            policy_model_name="openai/local-instruct", eval_model_name="openai/local-instruct",
            search_args={"n_actions": 1, "terminate_constraints": ["step_signal"]},
            component_args={"seed": 114, "max_new_tokens": 16, "verifier_max_new_tokens": 1536},
        )
        transition, policy, verifier = create_components(
            "language_grounded", "math_qa", model, model, None, None, config,
        )
        if not isinstance(policy, ContinuationPolicy) or not isinstance(verifier, PrefixVerifier):
            raise RuntimeError("Factory returned the wrong components")
        chunks = policy.get_actions([], query="Reply with OK only.", query_idx=0)
        state, _ = transition.step([], chunks[0], "Reply with OK only.", query_idx=0)
        if not transition.is_terminal(state, "Reply with OK only.", query_idx=0):
            raise RuntimeError("Factory did not preserve completion signals")
        for index, answer in enumerate(("42", "43")):
            state = [ThoughtStep(action="15 + ")]
            candidate = ThoughtStep(action="27 = " + answer)
            score, details = verifier.fast_reward(state, candidate, query, index, from_phase="expand")
            save(args.output_dir / f"verification_{index}.json", {"score": score, **details})
            request, response = captured[-1]["request"], captured[-1]["response"]
            expected_prompt = tokenizer.apply_chat_template(
                build_prompt(query, "15 + 27 = " + answer), tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            if request["prompt"] != expected_prompt or details["prompt"] != expected_prompt:
                raise RuntimeError("Whole-prefix prompt differs from the reference")
            if (request["seed"] != 114 + index or request["logprobs"] != 1
                    or request["max_tokens"] != 1536 or "messages" in request):
                raise RuntimeError("Verifier parameters were not forwarded")
            choice = response["choices"][0]
            if (details["response"] != choice["text"] or details["logprobs"] != choice["logprobs"]
                    or details["finish_reason"] != choice["finish_reason"]
                    or details["output_tokens"] != response["usage"]["completion_tokens"]):
                raise RuntimeError("Verifier evidence was discarded")
            if ("<think>" not in details["response"] or "</think>" not in details["response"]
                    or choice["finish_reason"] != "stop"
                    or choice["logprobs"]["content"][-1]["id"] != tokenizer.eos_token_id):
                raise RuntimeError("Inspect saved evidence: thinking/EOS check failed")
            if score != compare_score(choice["logprobs"], reference_score):
                raise RuntimeError("Live score differs from saved-response scoring")
            if verifier.reward(state, candidate, fast_reward=score) != score:
                raise RuntimeError("Reward transformed the prefix score")
        if len(captured) != 3:
            raise RuntimeError("Unexpected inference during transition or reward reuse")
        breakpoint()  # inspect: scores, captured, score, details, config
    print(f"Passed saved/live scoring, factory, prompt and evidence checks: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--saved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE"))
    args = parser.parse_args()
    if args.live and not args.base_url:
        parser.error("Live checks require --base-url or OPENAI_API_BASE")
    check_verifier(args)
