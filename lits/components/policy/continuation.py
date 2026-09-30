"""Generate successive chunks within one open assistant response."""

from itertools import count

from ..base import Policy
from ..registry import register_policy
from ...lm.openai_completion import OpenAICompletionModel
from ...structures.qa import ContinuationStep


@register_policy("continuation", task_type="language_grounded")
class ContinuationPolicy(Policy):
    """Continue a rendered task prompt without inserting new turns or step labels.

    query is the complete benchmark-formatted user prompt. tokenizer must match
    the server's model/template. The initial template is cached for the current
    query; state contains only retained ContinuationStep chunks. The inherited
    task_prompt_spec holds the system instruction, loaded from PromptRegistry by
    task_name unless supplied explicitly. The policy places query in a separate
    user message. Without a system instruction, keep the single-user template.

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

    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        """Build from existing config dictionaries and base_model.tokenizer.

        The caller loads the server-matching tokenizer on the raw model before
        create_components(). n_actions comes from search_args; seed, chunk size
        and sampling settings come from component_args. No tokenizer is downloaded.
        Pass task_prompt_spec and task_name through the normal Policy constructor;
        dataset_name takes precedence over task_name for benchmark prompt lookup.
        """
        return cls(base_model, base_model.tokenizer, n_actions=search_args["n_actions"],
                   seed=component_args.get("seed", 0),
                   max_new_tokens=component_args.get("max_new_tokens") or 512,
                   temperature=component_args.get("temperature", 0.7),
                   top_p=component_args.get("top_p", 0.8),
                   top_k=component_args.get("top_k", 20),
                   min_p=component_args.get("min_p", 0.0),
                   task_prompt_spec=kwargs.get("task_prompt_spec"),
                   task_name=kwargs.get("dataset_name") or kwargs.get("task_name"))

    def __init__(self, base_model, tokenizer, *, n_actions: int, seed: int = 0,
                 max_new_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.8, top_k: int = 20, min_p: float = 0.0,
                 task_prompt_spec: str | None = None, task_name: str | None = None):
        """Set sampling parameters, initial seed and the inherited prompt settings.

        task_prompt_spec optionally overrides the registered system instruction;
        an empty string explicitly omits it. query remains a separate user message.
        """
        if not isinstance(base_model, OpenAICompletionModel):
            raise TypeError("ContinuationPolicy requires OpenAICompletionModel")
        if tokenizer is None:
            raise ValueError("Attach the server-matching tokenizer to the raw model before construction")
        if n_actions < 1 or max_new_tokens < 1:
            raise ValueError("n_actions and max_new_tokens must be positive")
        super().__init__(base_model=base_model, task_prompt_spec=task_prompt_spec,
                         task_name=task_name, usr_prompt_spec="",
                         n_actions=n_actions, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p, top_k=top_k,
                         force_terminating_on_depth_limit=False)
        self.tokenizer = tokenizer
        self.min_p = min_p
        self._seeds = count(seed)
        self._query = None
        self._prompt = ""

    def _build_system_prompt(self) -> str:
        """Return the inherited system instruction for explicit template rendering."""
        return self.task_prompt_spec or ""

    def set_system_prompt(self) -> None:
        """Leave the raw model unchanged; the rendered prefix owns all instructions."""
        return

    def _is_at_depth_limit(self, state) -> bool:
        """Keep candidate count and temperature unchanged at every depth."""
        return False

    def _create_error_steps(self, n_actions: int, error_msg: str):
        """Propagate failure instead of creating apparently usable empty chunks."""
        raise RuntimeError(error_msg)

    def _build_messages(self, query: str) -> list[dict[str, str]]:
        """Keep the system instruction separate from the original query.

        No reference answer or retained chunks belong in these messages. Render
        them once, then append retained chunks within the open assistant response.
        """
        messages = []
        if self.task_prompt_spec:
            messages.append({"role": "system", "content": self._build_system_prompt()})
        return messages + [{"role": "user", "content": query}]

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
                self._build_messages(query), tokenize=False,
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
