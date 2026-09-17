"""Per-run OTLP file exporters and archive completion/failure metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


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
