"""Sequential checks for jev.py::JevPRM.fast_reward and threshold aggregation.

Run with PYTHONBREAKPOINT=0 to skip inspection. Optional --saved-replay reads
the earlier correctness replay; --output-dir saves fresh two-score responses.
Only saved question/reasoning inputs are transmitted, never reference labels.
"""

import argparse
import json
from pathlib import Path

from lits.components.factory import JevPRM
from lits.components.registry import ComponentRegistry


def main():
    """Check local aggregation, then optionally score saved candidates with the live API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-replay", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    prm = JevPRM.from_config(None, {}, {})
    if ComponentRegistry.get_reward_model("jev") is not JevPRM:
        raise RuntimeError("Reward registration failed")
    if prm.MODEL_ID != "jev-1.14.0":
        raise RuntimeError("JEV requests are not pinned to version 1.14.0")
    for correctness, usefulness, expected in ((0.7, 0.8, 0), (0.71, 0.8, 0.8), (1, 0, 0)):
        reward = prm.aggregate_scores(correctness, usefulness)
        if reward != expected or prm.reward([], "candidate", reward) != expected:
            raise RuntimeError("Threshold or reward reuse failed")
    for invalid in (-0.1, 1.1, float("nan"), True):
        try:
            prm.aggregate_scores(invalid, 0.5)
        except ValueError:
            continue
        raise RuntimeError("Invalid score accepted")
    print("Passed pinned-version, registry, strict-threshold, reward-reuse, and score-validation checks")
    if args.saved_replay is None:
        return
    if args.output_dir is None:
        parser.error("--saved-replay requires --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "records.jsonl").open("x") as output:
        for line in args.saved_replay.read_text().splitlines():
            saved = json.loads(line)
            state = saved["request"]["state"]
            reward, details = prm.fast_reward([], state["reasoning"], state["context"], saved["query_idx"])
            record = dict(query_idx=saved["query_idx"], candidate_id=saved["candidate_id"],
                          source=str(args.saved_replay), reward=reward, **details)
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps({key: record[key] for key in
                              ("query_idx", "candidate_id", "correctness_score", "usefulness", "reward", "seconds")}))
    breakpoint()  # inspect: prm, record, args.output_dir


if __name__ == "__main__":
    main()
