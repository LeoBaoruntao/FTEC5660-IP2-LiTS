"""Generate successive chunks within one open assistant response."""

from itertools import count

from ..base import Policy
from ...lm.openai_completion import OpenAICompletionModel
from ...structures.qa import ContinuationStep


class ContinuationPolicy(Policy):
    """Continue a rendered task prompt without inserting new turns or step labels.

    query is the complete benchmark-formatted user prompt. tokenizer must match
    the server's model/template. The initial non-thinking template is cached for
    the current query; state contains only retained ContinuationStep chunks.

    Each candidate uses a separate n=1 request and the next seed in an ordered
    sequence beginning at seed. Create a policy per run to reset that sequence;
    individual seeds are saved on steps. Depth does not change sampling or force
    termination. Search owns cumulative token limits and candidate deduplication.

    Example::

        policy = ContinuationPolicy(model, tokenizer, n_actions=2, seed=7)
        candidates = policy.get_actions([], query=task_prompt, query_idx=0)
        next_candidates = policy.get_actions([candidates[0]], query=task_prompt)
    """

    TASK_TYPE = "language_grounded"

    def __init__(self, base_model, tokenizer, *, n_actions: int, seed: int = 0,
                 max_new_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.8, top_k: int = 20, min_p: float = 0.0):
        """Set fixed sampling parameters and the starting per-request seed."""
        if not isinstance(base_model, OpenAICompletionModel):
            raise TypeError("ContinuationPolicy requires OpenAICompletionModel")
        if n_actions < 1 or max_new_tokens < 1:
            raise ValueError("n_actions and max_new_tokens must be positive")
        super().__init__(base_model=base_model, task_prompt_spec="", usr_prompt_spec="",
                         n_actions=n_actions, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p, top_k=top_k,
                         force_terminating_on_depth_limit=False)
        self.tokenizer = tokenizer
        self.min_p = min_p
        self._seeds = count(seed)
        self._query = None
        self._prompt = ""

    def _build_system_prompt(self) -> str:
        """No separate system instruction: query already contains the task prompt."""
        return ""

    def set_system_prompt(self) -> None:
        """Leave the raw model unchanged; the rendered prefix owns all instructions."""
        return

    def _is_at_depth_limit(self, state) -> bool:
        """Keep candidate count and temperature unchanged at every depth."""
        return False

    def _create_error_steps(self, n_actions: int, error_msg: str):
        """Propagate failure instead of creating apparently usable empty chunks."""
        raise RuntimeError(error_msg)

    def _get_actions(self, state, n_actions: int, temperature: float,
                     query: str, existing_siblings=None, **kwargs) -> list[ContinuationStep]:
        """Append retained chunks and sample candidates in seed order.

        Sibling-aware prompting is unsupported because it would alter the prefix.
        An EOS-completed trajectory must not be expanded. API errors propagate.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a benchmark-formatted string")
        if n_actions < 1 or existing_siblings:
            raise ValueError("Require positive n_actions and no sibling-aware prompting")
        if any(not isinstance(step, ContinuationStep) for step in state):
            raise TypeError("state must contain ContinuationStep chunks")
        if any(step.terminate for step in state):
            raise ValueError("Cannot continue a naturally completed trajectory")
        if query != self._query:
            self._prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            self._query = query
        prompt = self._prompt + "".join(step.action for step in state)
        candidates = []
        for _ in range(n_actions):
            seed = next(self._seeds)
            output = self._call_model(prompt, seed=seed, temperature=temperature,
                                      min_p=self.min_p)
            candidates.append(ContinuationStep.from_completion(output, seed))
        return candidates
