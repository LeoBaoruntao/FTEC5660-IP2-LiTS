"""Per-run OTLP file exporters, completion metadata, and offline archive selection."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


def filter_archives(
    source_root: str | Path, run_ids: Iterable[str], output_dir: str | Path,
) -> list[Path]:
    """Copy explicitly selected complete runs into a new, separate archive root.

    Args:
        source_root: Root against which the saved relative-path run IDs resolve.
        run_ids: Nonempty, unique POSIX paths matching each archive.json run_id.
            Unknown runs, incomplete/open archives, and unsafe paths are rejected.
        output_dir: New destination outside source_root, not an ancestor of it.
            Existing destinations are never overwritten or appended to.

    Returns:
        Copied run directories in the requested order, retaining their hierarchy::

            run_id = "math500/beam_llm/run_0/telemetry/example-42-repeat-0-attempt-0"
            copied = filter_archives("results", [run_id], "selected")
            # copied == [Path("selected").resolve() / run_id]

    Copy only traces.jsonl, logs.jsonl, and archive.json, byte-for-byte. Resource
    run IDs, span/log linkage, timestamps, and attributes stay unchanged even
    after relocation. No SDK, network, inference, or correctness evaluation runs.
    Callers choose run IDs from their own evaluation results; "complete" refers
    to telemetry, not answer correctness. OTLP payloads are not parsed or repaired.

    Validate all markers and regular files before creating output. A copy failure
    removes only the new destination. FileNotFoundError reports unknown paths;
    FileExistsError reports an existing destination; ValueError reports invalid
    selection, identity, status, or layout. Filesystem errors propagate.
    """
    source_root = Path(source_root).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir = output_dir.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError("Source and output roots must not overlap")
    if isinstance(run_ids, str):
        raise ValueError("Pass a collection of run IDs, not one string")
    run_ids = tuple(run_ids)
    if not run_ids:
        raise ValueError("Select at least one run ID")

    filenames = ("traces.jsonl", "logs.jsonl", "archive.json")
    selected, seen = [], set()
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Run IDs must be nonempty strings")
        relative = PurePosixPath(run_id)
        if (relative.is_absolute() or ".." in relative.parts or run_id == "."
                or relative.as_posix() != run_id or run_id in seen):
            raise ValueError(f"Unsafe or duplicate run ID: {run_id!r}")
        seen.add(run_id)
        directory = (source_root / relative).resolve(strict=True)
        if not directory.is_relative_to(source_root):
            raise ValueError(f"Run resolves outside source root: {run_id!r}")
        for filename in filenames:
            path = directory / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Archive requires a regular {filename}: {run_id!r}")
        metadata = json.loads((directory / "archive.json").read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or metadata.get("run_id") != run_id
                or metadata.get("status") != "complete" or metadata.get("error")):
            raise ValueError(f"Archive must be complete with matching run_id: {run_id!r}")
        selected.append((run_id, directory))

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for run_id, directory in selected:
            destination = output_dir / run_id
            destination.mkdir(parents=True, exist_ok=True)
            # Publish the completion marker only after both payload files exist.
            for filename in filenames:
                shutil.copyfile(directory / filename, destination / filename)
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return [output_dir / run_id for run_id in run_ids]


class _CheckedExporter:
    """Delegate to a real OTLP exporter and retain failures across later flushes."""

    def __init__(self, exporter: Any, failure_result: Any, archive: _Archive) -> None:
        """Wrap a span/log exporter with its failure enum and shared archive status."""
        self.exporter = exporter
        self.failure_result = failure_result
        self.archive = archive
        self.closed = False

    def export(self, batch: Any) -> Any:
        """Return the SDK result; failed writes must not interrupt application code."""
        try:
            result = self.exporter.export(batch)
            if result == self.failure_result:
                self.archive.fail("OTLP file export failed")
            return result
        except Exception as error:
            self.archive.fail(f"OTLP file export failed: {error}")
            return self.failure_result

    def shutdown(self) -> None:
        """Close once, recording errors without interrupting other exporters."""
        if not self.closed:
            self.closed = True
            try:
                self.exporter.shutdown()
            except Exception as error:
                self.archive.fail(f"OTLP file close failed: {error}")


class _Archive:
    """Own one run's file exporters and persist a sticky failure/completion status.

    The two telemetry pipelines use separate exporters::

        Spans       -> FileSpanExporter -> traces.jsonl
        Log records -> FileLogExporter  -> logs.jsonl
    """

    def __init__(self, directory: Path, run_id: str) -> None:
        """Create a new archive; refuse existing directories rather than append runs."""
        from opentelemetry.exporter.otlp.json.file import FileSpanExporter
        from opentelemetry.exporter.otlp.json.file._log_exporter import FileLogExporter
        from opentelemetry.sdk._logs.export import LogRecordExportResult
        from opentelemetry.sdk.trace.export import SpanExportResult

        self.directory = directory
        self.run_id = run_id
        self.error: str | None = None
        self.span_exporter: _CheckedExporter | None = None
        self.log_exporter: _CheckedExporter | None = None
        directory.mkdir(parents=True, exist_ok=False)
        try:
            if not self.write_status("open"):
                raise OSError(self.error)
            self.span_exporter = _CheckedExporter(
                FileSpanExporter(directory / "traces.jsonl"), SpanExportResult.FAILURE, self,
            )
            self.log_exporter = _CheckedExporter(
                FileLogExporter(directory / "logs.jsonl"), LogRecordExportResult.FAILURE, self,
            )
        except Exception as error:
            self.fail(f"Archive initialization failed: {error}")
            self.close_exporters()
            self.write_status("incomplete")
            raise

    def fail(self, message: str) -> None:
        """Keep the first error and report it outside telemetry to avoid export recursion."""
        if self.error is None:
            self.error = message
            try:
                print(f"LiTS telemetry archive incomplete ({self.directory}): {message}", file=sys.stderr)
            except (OSError, ValueError):
                pass

    def write_status(self, status: str) -> bool:
        """Atomically replace the marker; an earlier failure can never become complete."""
        payload = {"run_id": self.run_id, "status": "incomplete" if self.error else status}
        if self.error:
            payload["error"] = self.error
        try:
            temporary = self.directory / "archive.json.tmp"
            temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            temporary.replace(self.directory / "archive.json")
        except OSError as error:
            self.fail(f"Archive status write failed: {error}")
        return self.error is None

    def close_exporters(self) -> None:
        """Close both files, including after a partially failed SDK initialization."""
        for exporter in (self.span_exporter, self.log_exporter):
            if exporter is not None:
                exporter.shutdown()
