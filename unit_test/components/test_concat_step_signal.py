"""Sequential check for ``ConcatTransition`` step-based termination.

Run with ``python -m unit_test.components.test_concat_step_signal`` and inspect
each breakpoint. Set ``PYTHONBREAKPOINT=0`` for non-interactive execution.
"""

from lits import get_lm
from lits.components.transition.concat import ConcatTransition
from lits.structures import ThoughtStep


MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def check_concat_step_signal() -> None:
    """Check the contracts of ``concat.py::ConcatTransition`` sequentially."""
    base_model = get_lm(MODEL_NAME)

    default_transition = ConcatTransition.from_config(
        base_model=base_model,
        search_args={},
        component_args={},
    )
    print("default constraints:", default_transition.terminate_constraints)
    breakpoint()  # inspect: default_transition.terminate_constraints == ["binary_sampling"]

    transition = ConcatTransition(
        base_model=base_model,
        terminate_constraints=["step_signal"],
    )
    supplied_step = ThoughtStep(
        action="The answer is 42",
        error="inspection marker",
        terminate=True,
    )
    state, _ = transition.step([], supplied_step, "What is 6 times 7?")
    print("materialized step:", state[-1].to_dict())
    breakpoint()  # inspect: state[-1] preserves action, error, and terminate=True

    continuing_state = [ThoughtStep(action="Compute 6 times 7", terminate=False)]
    continuing = transition.is_terminal(continuing_state, "What is 6 times 7?")
    print("false signal:", continuing)
    breakpoint()  # inspect: continuing is False

    completed = transition.is_terminal(state, "What is 6 times 7?")
    print("true signal:", completed)
    breakpoint()  # inspect: completed is True


if __name__ == "__main__":
    check_concat_step_signal()
