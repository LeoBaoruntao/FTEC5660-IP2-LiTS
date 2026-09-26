"""Check mathematical JSON validation; optionally generate two real structured steps.

Run with PYTHONBREAKPOINT=0 python -m unit_test.components.test_concat_json_math.
Add --live for two local-instruct calls using OPENAI_API_BASE/OPENAI_API_KEY.
"""

import argparse
import json

from lits.components.policy.concat import ConcatPolicy


def check_math_json():
    """Exercise concat.py::ConcatPolicy._parse_structured_output without inference."""
    cases = [
        (r'{"step": "Use cos(theta)."}', True),
        (r'{"step": "Use \\cos(\\theta)."}', True),
        (r'{"step": "Use \cos(\theta)."}', False),
        (r'{"step": "Use \frac{1}{2}."}', False),
        (r'{"step": "Use \theta."}', False),
        (r'{"step": "Line one\nLine two"}', True),
        ('{"step": "unfinished', False),
        ('{"step": "x", "answer": "2"}', False),
    ]
    for raw, expected in cases:
        try:
            result = ConcatPolicy._parse_structured_output(raw)
        except ValueError:
            accepted = False
        else:
            accepted = True
            if result != json.loads(raw):
                raise RuntimeError("Valid mathematical text changed during parsing")
        if accepted != expected:
            raise RuntimeError(f"Unexpected validation result: {raw!r}")
    print(f"Passed {len(cases)} mathematical JSON checks")
    breakpoint()  # inspect: cases; malformed escapes must not become reasoning steps


def check_live_math_json():
    """Use concat.py::ConcatPolicy.get_actions with a saved prefix and plain-text query.

    No reference answers enter generation. Retain model usage and generated steps
    in a fresh temporary directory; existing search artifacts remain untouched.
    """
    from pathlib import Path
    from tempfile import mkdtemp
    from lits import get_lm
    from lits.lm import setup_inference_logging
    from lits.structures import ThoughtStep

    output = Path(mkdtemp(prefix="lits-concat-json-math-"))
    model = get_lm("openai/local-instruct", enable_thinking=False)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=output)
    policy = ConcatPolicy.from_config(
        base_model=model, search_args={"n_actions": 1},
        component_args={"output_format": "structured", "max_new_tokens": 1024},
        task_name="scibench_calculus",
    )
    query = ("A fluid has density 870 kg/m^3 and velocity v = z i + y^2 j + x^2 k. "
             "Find the mass flow rate outward through x^2+y^2=4, 0 <= z <= 1, in kg/s.")
    state = [ThoughtStep(action=(
        "Compute the flux of the fluid through the surface of the cylinder using the "
        "formula for mass flow rate, which is the integral of the dot product of the "
        "velocity vector and the normal vector over the surface, multiplied by the "
        "density of the fluid."))]
    results = []
    for index in range(2):
        steps = policy.get_actions(state=state, n_actions=1, query=query,
                                   query_idx=index, from_phase="expand")
        results.append([step.to_dict() for step in steps])
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Live results: {output}\n{json.dumps(results, indent=2)}")
    breakpoint()  # inspect: results; valid format does not establish mathematical correctness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    check_math_json()
    if args.live:
        check_live_math_json()
