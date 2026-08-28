"""Sequential check for structured ``ConcatPolicy`` output.

Run with ``python -m unit_test.components.test_concat_structured_policy``.
Set ``PYTHONBREAKPOINT=0`` for non-interactive execution. The configured model
is called once.
"""

import os

from lits import get_lm
from lits.components.policy.concat import ConcatPolicy
from lits.structures import ThoughtStep


MODEL_NAME = os.environ.get(
    "LITS_TEST_MODEL_NAME",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
)


def check_structured_concat_policy() -> None:
    """Inspect one real call through ``concat.py::ConcatPolicy.get_actions``."""
    model_kwargs = {}
    if base_url := os.environ.get("LITS_TEST_BASE_URL"):
        model_kwargs["base_url"] = base_url
        model_kwargs["api_key"] = os.environ.get("LITS_TEST_API_KEY", "no-key")
    base_model = get_lm(MODEL_NAME, **model_kwargs)
    policy = ConcatPolicy.from_config(
        base_model=base_model,
        search_args={"n_actions": 1},
        component_args={"output_format": "structured", "max_new_tokens": 128},
        task_name="math500",
    )
    state = [ThoughtStep(action="Six multiplied by seven equals 42.")]
    steps = policy.get_actions(
        state=state,
        n_actions=1,
        query="What is 6 multiplied by 7?",
        query_idx=0,
        from_phase="expand",
    )
    print("structured step:", steps[0].to_dict())
    breakpoint()  # inspect: one ThoughtStep with canonical action and terminate


if __name__ == "__main__":
    check_structured_concat_policy()
