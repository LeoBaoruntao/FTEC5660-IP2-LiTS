"""Thinking-enabled verification of a whole reasoning prefix using token probabilities."""

import math
from itertools import count

from ..base import RewardModel
from ..registry import register_reward_model
from ...lm.openai_completion import OpenAICompletionModel
from ...structures import ThoughtStep


VERIFIER_INSTRUCTION = (
    "You are a careful and precise verifier. "
    "Your task is to assess whether the assistant's reasoning so far is logically and mathematically correct. "
    "You must not attempt to solve the problem yourself or continue the reasoning. "
    "Only check whether the existing reasoning is correct step by step. "
    "At the end of your response, on a new line, write exactly:\n"
    '"Is the existing reasoning correct (Yes/No)? X"\n'
    "where X is either Yes or No. Only write Yes if all steps so far are logically correct."
)


@register_reward_model("prefix_verifier", task_type="language_grounded")
class PrefixVerifier(RewardModel):
    """Score the complete retained prefix, not correctness/usefulness separately.

    One raw completion uses the tokenizer's thinking-enabled chat template.
    The penultimate logprob position supplies the score: positive probability for
    Yes, negative for No, -1 for another token. No verdict is searched for in prose.
    fast_reward returns (score, details) retaining prompt, full unstripped response,
    finish reason, token evidence, usage and seed. reward reuses that score.

    Requires the verified llama.cpp logprobs.content format. This positional rule
    assumes the final position is EOS for natural completions. Cutoffs retain their
    finish reason but use the same positional rule; they are not parsed as verdicts.
    """

    TASK_TYPE = "language_grounded"

    def _get_llm_role(self) -> str:
        """Use the supported PRM role for verifier inference accounting."""
        return "prm_language"

    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        """Use the raw model's tokenizer and explicitly configured verifier limit.

        Set verifier_max_new_tokens in component_args; no experimental budget is
        guessed. Sampling settings and initial seed use the same keys as the policy.
        """
        limit = component_args.get("verifier_max_new_tokens")
        if limit is None:
            raise ValueError("Set component_args['verifier_max_new_tokens'] explicitly")
        return cls(base_model, base_model.tokenizer, max_new_tokens=limit,
                   seed=component_args.get("seed", 0),
                   temperature=component_args.get("temperature", 0.7),
                   top_p=component_args.get("top_p", 0.8),
                   top_k=component_args.get("top_k", 20),
                   min_p=component_args.get("min_p", 0.0))

    def __init__(self, base_model, tokenizer, *, max_new_tokens: int, seed: int = 0,
                 temperature: float = 0.7, top_p: float = 0.8,
                 top_k: int = 20, min_p: float = 0.0):
        """Initialize a raw verifier; caller supplies the matching tokenizer and budget."""
        if not isinstance(base_model, OpenAICompletionModel):
            raise TypeError("PrefixVerifier requires OpenAICompletionModel")
        if tokenizer is None or max_new_tokens < 1:
            raise ValueError("Require a server-matching tokenizer and positive verifier token limit")
        super().__init__(base_model=base_model, task_prompt_spec="", usr_prompt_spec="",
                         max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, top_k=top_k)
        self.tokenizer = tokenizer
        self.min_p = min_p
        self._seeds = count(seed)

    @staticmethod
    def verification_score(logprobs: dict) -> tuple[float, dict]:
        """Read the last alternative at the penultimate position, preserving order.

        With logprobs=1, expect one or two alternatives, as in the source scorer.
        Return the score and selected token evidence. Missing/malformed logprobs
        raise ValueError rather than fabricating a score. Unknown tokens score -1.
        """
        positions = (logprobs or {}).get("content", [])
        if len(positions) < 2:
            raise ValueError("Verifier requires at least two logprob positions")
        alternatives = positions[-2].get("top_logprobs", [])
        if not 1 <= len(alternatives) <= 2:
            raise ValueError("Expected one or two alternatives at the penultimate position")
        selected = alternatives[-1]
        token, logprob = selected.get("token"), selected.get("logprob")
        if (not isinstance(token, str) or type(logprob) not in (int, float)
                or not math.isfinite(logprob) or logprob > 0):
            raise ValueError("Invalid verifier token or log probability")
        score = math.exp(logprob) if "yes" in token.lower() else (
            -math.exp(logprob) if "no" in token.lower() else -1.0)
        return score, selected

    def _fast_reward(self, state, action_or_step, query, query_idx, from_phase=""):
        """Verify retained state plus candidate as one assistant response."""
        action = action_or_step.action if isinstance(action_or_step, ThoughtStep) else action_or_step
        prefix = "".join(step.action for step in state) + action
        messages = [{"role": "user", "content": query},
                    {"role": "assistant", "content": prefix},
                    {"role": "user", "content": VERIFIER_INSTRUCTION}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        seed = next(self._seeds)
        output = self._call_model(prompt, seed=seed, min_p=self.min_p, logprobs=1)
        score, token = self.verification_score(output.logprobs)
        return score, {"prompt": prompt, "response": output.text,
                       "finish_reason": output.finish_reason, "verdict_token": token,
                       "logprobs": output.logprobs, "seed": seed,
                       "input_tokens": output.input_tokens, "output_tokens": output.output_tokens}

    def calculate_reward(self, fast_reward: float) -> float:
        """Preserve the signed probability without scaling or correctness gating."""
        return fast_reward

    def reward(self, state, action, fast_reward: float, **kwargs) -> float:
        """Reuse the supplied whole-prefix score without further inference."""
        return fast_reward
