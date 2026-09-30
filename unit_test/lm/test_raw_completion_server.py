"""Probe a real llama.cpp server before adding raw completion to LiTS.

Run from the repository root with OPENAI_API_BASE set to the server's /v1 URL::

    PYTHONBREAKPOINT=0 python -m unit_test.lm.test_raw_completion_server \
        --output-dir /path/to/new/persistent/evidence

Requires the pinned Qwen tokenizer in the local Hugging Face cache. At most
3137 output tokens are requested, with no automatic retries or benchmark runs.
Each request/response is saved without authentication headers. Existing output
directories are rejected. Failures remain inspectable in summary.json.

Compare openai_chat.py::OpenAIChatModel.__call__ (which formats chat messages)
with raw completion, and llm_as_verifier.py::verification_score in the supplied
reference implementation (which reads the penultimate token's log probability).
This tests protocol behavior, not benchmark accuracy or full-context quality.

Add --existing-lits to compare openai_chat.py::OpenAIChatModel.__call__ with
its SDK client's raw completion on the same prefix, without changing production
code. This mode requests at most 1584 output tokens and captures real HTTP traffic.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


TOKENIZER_REVISION = "a80f5e57cce20e57b65145f4213844dec1a80834"
VERIFIER_INSTRUCTION = (
    "You are a careful and precise verifier. "
    "Your task is to assess whether the assistant's reasoning so far is logically and mathematically correct. "
    "You must not attempt to solve the problem yourself or continue the reasoning. "
    "Only check whether the existing reasoning is correct step by step. "
    "At the end of your response, on a new line, write exactly:\n"
    '"Is the existing reasoning correct (Yes/No)? X"\n'
    "where X is either Yes or No. Only write Yes if all steps so far are logically correct."
)


def save(path, value):
    """Save JSON evidence; callers must use a fresh output directory."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_server(base_url, model, output_dir):
    """Run bounded real requests and retain failures before reporting them.

    base_url ends in /v1; model is the serving alias. output_dir must not exist.
    Token arrays are compared exactly, including whitespace and special tokens.
    A same-seed replay is an observation, not a cross-backend determinism claim.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"base_url": base_url, "model": model, "tokenizer_revision": TOKENIZER_REVISION,
               "checks": {}, "observations": {}}
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    root_url = base_url.removesuffix("/v1")

    def request(name, endpoint, payload=None):
        """Save request and raw response, including HTTP or transport errors."""
        record = {"endpoint": endpoint, "request": payload}
        started = time.monotonic()
        try:
            response = client.request("GET" if payload is None else "POST",
                                      root_url + endpoint, json=payload)
            record.update(status=response.status_code, response_text=response.text)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            save(output_dir / f"{name}.json", record)

    def check(name, passed):
        """Retain every check, without aborting subsequent diagnostic probes."""
        summary["checks"][name] = bool(passed)
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    def render(name, messages, thinking):
        """Compare the pinned client template and tokenization with the server."""
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=thinking)
        server = request(name + "_template", "/apply-template", {
            "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        })
        check(name + "_template", prompt == server["prompt"])
        encoded = request(name + "_tokens", "/tokenize", {
            "content": prompt, "add_special": True, "parse_special": True,
        })
        check(name + "_tokens", encoded["tokens"] == tokenizer.encode(prompt, add_special_tokens=False))
        save(output_dir / f"{name}_client.json", {"messages": messages, "prompt": prompt,
                                                 "enable_thinking": thinking})
        return prompt

    def complete(name, prompt, seed, limit):
        """Send one raw completion; retain text and all logprobs without stripping."""
        response = request(name, "/v1/completions", {
            "model": model, "prompt": prompt, "n": 1, "seed": seed,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "max_tokens": limit, "logprobs": 1, "cache_prompt": False,
        })
        check(name + "_single_choice", len(response["choices"]) == 1)
        choice = response["choices"][0]
        check(name + "_prompt_tokens", response["usage"]["prompt_tokens"] ==
              len(tokenizer.encode(prompt, add_special_tokens=False)))
        check(name + "_finish_reason", choice["finish_reason"] in ("length", "stop"))
        return choice

    with httpx.Client(timeout=120, headers={
        "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "no-key"),
    }) as client:
        try:
            props = request("properties", "/props")
            models = request("models", "/v1/models")
            check("model_alias", any(item["id"] == model for item in models["data"]))
            summary["observations"].update(
                build=props.get("build_info"), model_path=props.get("model_path"),
                context=props["default_generation_settings"]["n_ctx"],
                client_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            )
            messages = [{"role": "user", "content": "List integers 1 through 100 separated by commas only."}]
            prompt = render("generator", messages, False)
            first = complete("chunk_1", prompt, 111, 16)
            second = complete("chunk_2", prompt + first["text"], 112, 16)
            check("chunk_boundary", first["finish_reason"] == "length")
            check("no_new_message", all("<|im_start|>" not in c["text"] for c in (first, second)))
            replay = complete("seed_replay", prompt, 111, 16)
            alternative = complete("seed_alternative", prompt, 112, 16)
            summary["observations"]["same_seed_same_text"] = first["text"] == replay["text"]
            summary["observations"]["different_seed_same_text"] = first["text"] == alternative["text"]
            native = request("seed_and_whitespace", "/completion", {
                "prompt": prompt + first["text"] + " \n\n", "seed": 113,
                "n_predict": 1, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "min_p": 0.0, "cache_prompt": False,
            })
            check("native_prompt_whitespace", native["prompt"] == prompt + first["text"] + " \n\n")
            settings = native["generation_settings"]
            check("seed_forwarding", settings["seed"] == 113)
            check("sampling_parameters", settings["top_k"] == 20 and
                  math.isclose(settings["temperature"], 0.7, abs_tol=1e-6) and
                  math.isclose(settings["top_p"], 0.8, abs_tol=1e-6) and settings["min_p"] == 0)

            for label, answer in (("Yes", "15 + 27 = 42"), ("No", "15 + 27 = 43")):
                messages = [{"role": "user", "content": "What is the sum of 15 and 27?"},
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content": VERIFIER_INSTRUCTION}]
                prompt = render("verifier_" + label, messages, True)
                choice = complete("verification_" + label, prompt, 114, 1536)
                entries = choice["logprobs"]["content"]
                tokens = [entry["token"] for entry in entries]
                probabilities = [entry["logprob"] for entry in entries]
                check(label + "_thinking", "<think>" in choice["text"] and "</think>" in choice["text"])
                check(label + "_natural_stop", choice["finish_reason"] == "stop")
                # This server returns EOS with empty text; its token ID is authoritative.
                check(label + "_eos", bool(entries) and entries[-1]["id"] == tokenizer.eos_token_id)
                selected = tokens[-2] if len(tokens) >= 2 else ""
                check(label + "_penultimate_verdict", selected.strip().lower() == label.lower())
                score = None
                if len(probabilities) >= 2 and probabilities[-2] is not None:
                    score = math.exp(probabilities[-2]) if "yes" in selected.lower() else (
                        -math.exp(probabilities[-2]) if "no" in selected.lower() else -1)
                check(label + "_finite_logprobs", bool(probabilities) and
                      all(value is not None and math.isfinite(value) for value in probabilities))
                summary["observations"][label] = {"last_tokens": entries[-8:], "score": score,
                                                   "finish_reason": choice["finish_reason"]}
            breakpoint()  # inspect: summary, first, second, choice, tokens, probabilities
        except Exception as error:
            summary["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save(output_dir / "summary.json", summary)
            print(f"Evidence: {output_dir}", flush=True)
    failures = [name for name, passed in summary["checks"].items() if not passed]
    if failures:
        raise RuntimeError("Compatibility checks failed: " + ", ".join(failures))


def check_existing_lits(base_url, model_name, output_dir):
    """Expose existing wrapper gaps using real calls, not substituted responses.

    Compare an already-rendered string, assistant-ending messages, and the
    thinking verifier. HTTP response hooks retain what the wrapper receives;
    returned Output attributes show what its callers can actually use.
    """
    from lits import get_lm

    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", revision=TOKENIZER_REVISION, local_files_only=True,
    )
    messages = [{"role": "user", "content": "List integers 1 through 100 separated by commas only."}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False) + "1, 2, "
    observations = {}
    captured = []

    def capture(response):
        """Record real HTTP bodies without authentication headers."""
        response.read()
        record = {"url": str(response.request.url),
                  "request": json.loads(response.request.content),
                  "status": response.status_code, "response_text": response.text}
        captured.append(record)
        save(output_dir / f"http_{len(captured)}.json", record)

    model = get_lm("openai/" + model_name, base_url=base_url, enable_thinking=False)
    with httpx.Client(event_hooks={"response": [capture]}, timeout=120) as http_client:
        model.client = model.client.with_options(http_client=http_client, timeout=120, max_retries=0)
        try:
            raw = model.client.completions.create(
                model=model_name, prompt=prefix, max_tokens=16, temperature=0.7,
                top_p=0.8, seed=111, logprobs=1,
                extra_body={"top_k": 20, "min_p": 0.0},
            )
            observations["raw_sdk"] = raw.model_dump()
            verifier = [{"role": "user", "content": "What is the sum of 15 and 27?"},
                        {"role": "assistant", "content": "15 + 27 = 42"},
                        {"role": "user", "content": VERIFIER_INSTRUCTION}]
            cases = [("rendered_string", prefix, False, 16),
                     ("assistant_prefix", messages + [{"role": "assistant", "content": "1, 2, "}], False, 16),
                     ("thinking_verifier", verifier, True, 1536)]
            for name, prompt, thinking, limit in cases:
                model.enable_thinking = thinking
                result = model(prompt, temperature=0.7, top_p=0.8, max_new_tokens=limit,
                               seed=111, top_k=20, min_p=0.0, logprobs=True)
                wire = json.loads(captured[-1]["response_text"])
                choice = wire["choices"][0]
                content = choice["message"].get("content") or ""
                observations[name] = {
                    "sent_request": captured[-1]["request"],
                    "returned_output": vars(result), "wire_finish_reason": choice["finish_reason"],
                    "wire_logprobs": choice.get("logprobs"),
                    "wire_reasoning_content": choice["message"].get("reasoning_content"),
                    "wire_text": content, "whitespace_removed": content != result.text,
                    "prompt_tokens": wire["usage"]["prompt_tokens"],
                }
                print(name + ": " + json.dumps({
                    "returned_fields": list(vars(result)),
                    "wire_finish_reason": choice["finish_reason"],
                    "whitespace_removed": content != result.text,
                    "forwarded_parameters": sorted(captured[-1]["request"]),
                }), flush=True)
            breakpoint()  # inspect: observations, captured; compare actual wire data with Output
        finally:
            save(output_dir / "existing_lits.json", observations)
            print(f"Evidence: {output_dir}", flush=True)


def main():
    """Require an explicit persistent evidence directory before live requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--model", default="local-instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-lits", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not args.base_url.rstrip("/").endswith("/v1"):
        parser.error("Set OPENAI_API_BASE or --base-url to a URL ending in /v1")
    check = check_existing_lits if args.existing_lits else check_server
    check(args.base_url.rstrip("/"), args.model, args.output_dir)


if __name__ == "__main__":
    main()
