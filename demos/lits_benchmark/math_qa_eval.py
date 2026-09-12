"""Registered numeric QA correctness checking and numeric/yes-no comparison helpers."""

from lits.benchmarks.registry import register_evaluator
from lits.components.utils import strip_num


@register_evaluator("math500")
@register_evaluator("gsm8k")
def evaluate_math_qa(predicted, ground_truth) -> bool:
    """Compare a predicted answer with a numeric QA reference without model calls.

    Uses eval_output's numeric normalization, with exact-string fallback on
    parsing errors. Accepts an extracted prediction and a reference convertible
    to str; returns correctness only. Answer extraction and LLM judging belong
    to the caller.
    """
    try:
        return eval_output(str(ground_truth), predicted, type="number_exact")
    except (AssertionError, ValueError):
        return predicted == str(ground_truth)


def eval_output(answer, output, type="number_exact"):
    """Evaluate the output of a model against the expected answer.

    Args:
        answer (str): The expected answer.
        output (str): The model's output.
        type (str, optional): The evaluation type. Available types are:
            * "number_exact" (exact match of numbers).
            * "yn" (yes/no answer).

    Returns:
        bool: True if the output is correct, False otherwise.
    """
    # pre-check whether an answer exists
    assert output is not None, "Output is None"
    assert type in ["number_exact", "yn"], f"Unknown evaluation type: {type}"
    answer = strip_num(answer)
    if output == "":
        return False

    # exact match
    if type == "number_exact":
        output, answer = normalize_number_pair(output, answer)
    elif type == "yn":
        output = normalize_yn(output)
        answer = normalize_yn(answer)
    else:
        raise ValueError(f"Unknown evaluation type: {type}")
    return output == answer


def normalize_number_pair(output, answer):
    """ Normalize the output and answer to a number pair """
    answer = answer.replace(",", "")
    try:
        output = int(output)
        answer = int(answer)
        return output, answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output, answer
    except ValueError:
        pass
    return output, answer


def normalize_yn(text):
    """Normalize yes/no aliases, leaving other lowercased text unchanged."""
    text = text.lower()
    if text in ["yes", "y", "true"]:
        text = "yes"
    elif text in ["no", "n", "false"]:
        text = "no"
    return text
