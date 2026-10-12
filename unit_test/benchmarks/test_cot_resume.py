"""Exercise cot.py::run_cot resume persistence without models or inference.

Run PYTHONBREAKPOINT=0 python -m unit_test.benchmarks.test_cot_resume.
Deterministic callbacks exercise storage and dispatch through the real registry;
no model is mocked. Temporary runs contain successes, failures and missing IDs.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lits.benchmarks import register_dataset
from lits.benchmarks.cot import load_resume_records, run_cot


@register_dataset("cot_resume_check")
def load_cohort():
    """Return a small local cohort for execution/persistence checks."""
    return [{"question": "First question"}, {"question": "Second question"}]


def main():
    """Check resume, provenance, validation, logs, atomic saves and cancellation."""
    with TemporaryDirectory(prefix="cot-resume-") as temporary:
        directory = Path(temporary) / "run"
        config = dict(dataset="cot_resume_check", output_dir=str(directory),
                      samples=2, seed=7, temperature=0.7)
        attempted = []

        def generate(question, index, repeat):
            """Record dispatch and return a deterministic persistence payload."""
            attempted.append((index, repeat))
            return dict(raw_response=question, seed=7 + index + repeat)

        run_cot(config, None, generate)
        path = directory / "responses.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        broken = [original[0], dict(query_idx=0, sample_index=1, error="HTTP 500"), original[2]]
        path.write_text("".join(json.dumps(row) + "\n" for row in broken))
        before = path.read_bytes()
        metadata = (directory / "config.json").read_bytes()
        for name in ("execution.log", "inferencelogger.log"):
            with (directory / name).open("a") as stream:
                stream.write('{"role": "resume-log-sentinel"}\n')
        attempted.clear()
        resumed = {**config, "resume": True}
        run_cot(resumed, None, generate)
        result = [json.loads(line) for line in path.read_text().splitlines()]
        if attempted != [(0, 1), (1, 1)] or len(result) != 4:
            raise RuntimeError("Resume did not generate exactly the failed and missing samples")
        if result[0] != original[0] or result[2] != original[2]:
            raise RuntimeError("Successful response, timing or cost was modified")
        if [r["seed"] for r in result] != [7, 8, 8, 9] or any("error" in r for r in result):
            raise RuntimeError("Seed/identity changed or failed records were not replaced")
        backups = list(directory.glob("responses.before-resume-*.jsonl"))
        if len(backups) != 1 or backups[0].read_bytes() != before:
            raise RuntimeError("Previous responses were not backed up intact")
        if (directory / "config.json").read_bytes() != metadata:
            raise RuntimeError("Original configuration was overwritten")
        for name in ("execution.log", "inferencelogger.log"):
            if "resume-log-sentinel" not in (directory / name).read_text():
                raise RuntimeError(f"Resume overwrote {name}")
        attempted.clear()
        completed = path.read_bytes()
        run_cot(resumed, None, generate)
        if attempted or path.read_bytes() != completed:
            raise RuntimeError("Completed run was regenerated or rewritten")
        try:
            run_cot({**resumed, "temperature": 0.6}, None, generate)
        except ValueError:
            pass
        else:
            raise RuntimeError("Changed generation settings were accepted")
        try:
            run_cot(config, None, generate)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("Existing directory was accepted without resume")
        saved_config = json.loads(metadata)
        cohort = json.loads((directory / "cohort.json").read_text())
        for invalid in (result + [result[0]], [dict(query_idx=0, sample_index=9, error="bad")]):
            path.write_text("".join(json.dumps(row) + "\n" for row in invalid))
            try:
                load_resume_records(directory, saved_config, cohort)
            except ValueError:
                pass
            else:
                raise RuntimeError("Duplicate/out-of-range IDs were accepted")
        path.write_bytes(completed)
        try:
            load_resume_records(directory, saved_config, [{"query_idx": 0, "example": {}}])
        except ValueError:
            pass
        else:
            raise RuntimeError("Changed cohort was accepted")

        def interrupt(question, index, repeat):
            """Persist one new sample, then simulate a recovery-budget stop."""
            if (index, repeat) == (1, 1):
                raise SystemExit("Recovery exhausted")
            return generate(question, index, repeat)

        path.write_text("".join(json.dumps(row) + "\n" for row in broken))
        try:
            run_cot(resumed, None, interrupt)
        except SystemExit:
            pass
        else:
            raise RuntimeError("SystemExit did not terminate resume")
        partial = [json.loads(line) for line in path.read_text().splitlines()]
        if len(partial) != 3 or "error" in partial[1] or list(directory.glob(".responses-*")):
            raise RuntimeError("Interrupted resume lost its last atomic checkpoint")
        attempted.clear()
        run_cot(resumed, None, generate)
        if attempted != [(1, 1)]:
            raise RuntimeError("Repeated resume did not skip its previous successful repair")
        print("Passed resume identity, backup, logs, settings, validation and interruption checks")
        breakpoint()  # inspect: result, attempted, backups, directory


if __name__ == "__main__":
    main()
