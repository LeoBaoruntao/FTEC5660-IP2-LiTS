"""Per-run OTLP archives, offline selection, and explicitly requested Logfire uploads."""

from __future__ import annotations

import base64
import json
import math
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


def _complete_archive(directory: Path) -> str:
    """Validate required regular files and the completion marker; return its run ID."""
    for filename in ("traces.jsonl", "logs.jsonl", "archive.json"):
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Archive requires a regular {filename}: {directory}")
    metadata = json.loads((directory / "archive.json").read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("run_id"), str)
            or not metadata["run_id"] or metadata.get("status") != "complete"
            or metadata.get("error")):
        raise ValueError(f"Archive must be complete with a run_id: {directory}")
    return metadata["run_id"]


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
        if _complete_archive(directory) != run_id:
            raise ValueError(f"Archive run_id does not match selection: {run_id!r}")
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


def _encode_archive_batch(payload: dict, signal: str, run_id: str) -> tuple[bytes, int]:
    """Encode OTLP JSON as protobuf without changing IDs, timestamps, or attributes."""
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    if signal == "traces":
        resource_key, scope_key, record_key = "resourceSpans", "scopeSpans", "spans"
        message = ExportTraceServiceRequest()
    else:
        resource_key, scope_key, record_key = "resourceLogs", "scopeLogs", "logRecords"
        message = ExportLogsServiceRequest()
    count = 0
    for resource in payload[resource_key]:
        attributes = resource.get("resource", {}).get("attributes", [])
        identities = [item["value"] for item in attributes if item["key"] == "lits.run_id"]
        if identities != [{"stringValue": run_id}]:
            raise ValueError("OTLP resource run_id does not match archive.json")
        for scope in resource.get(scope_key, []):
            for record in scope.get(record_key, []):
                count += 1
                # OTLP JSON uses hex IDs; protobuf's generic JSON parser expects base64.
                for item in (record, *record.get("links", [])):
                    for key, size in (("traceId", 16), ("spanId", 8), ("parentSpanId", 8)):
                        value = item.get(key)
                        if not value:
                            continue
                        decoded = bytes.fromhex(value)
                        if len(value) != size * 2 or len(decoded) != size:
                            raise ValueError(f"Invalid OTLP {key}")
                        item[key] = base64.b64encode(decoded).decode("ascii")
    return ParseDict(payload, message).SerializeToString(), count


def upload_archives(
    directories: Iterable[str | Path], *, base_url: str, token: str, timeout: float = 30,
) -> dict[str, dict[str, int]]:
    """Explicitly upload selected complete archives to hosted Logfire over OTLP/HTTP.

    Args:
        directories: Run directories returned by filter_archives, including relocated
            copies. Only these directories are read; duplicate run IDs are rejected.
        base_url: Explicit https://logfire-us.pydantic.dev or EU equivalent.
        token: Explicit project write token. No credentials or endpoints are discovered.
        timeout: Positive finite timeout in seconds for each HTTP request.

    Returns:
        Per-run accepted record counts, e.g. {run_id: {"traces": 3, "logs": 1}}.

    Usage::

        selected = filter_archives("results", [run_id], "selected")
        counts = upload_archives(
            selected, base_url="https://logfire-us.pydantic.dev", token=write_token,
        )

    All markers, resource identities, and OTLP batches are validated and buffered
    before any network request. Empty signal files are allowed. IDs, timestamps,
    attributes, and native-log span links are preserved; files are never modified.
    This does not configure tracing, run inference, or enable future runtime uploads.
    Requires the observability dependencies only when called.

    Each nonempty JSONL batch is sent once, with no redirects or automatic retries.
    HTTP errors, partial rejection, or transport errors raise RuntimeError. Upload
    is not transactional: earlier batches may already exist remotely. Check the
    backend before retrying; this function does not deduplicate previous uploads.
    """
    if base_url not in ("https://logfire-us.pydantic.dev", "https://logfire-eu.pydantic.dev"):
        raise ValueError("Specify a hosted Logfire regional HTTPS base URL")
    if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
        raise ValueError("Supply an explicit project write token")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if isinstance(directories, (str, Path)):
        raise ValueError("Pass a collection of archive directories")

    batches, counts = [], {}
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        run_id = _complete_archive(directory)
        if run_id in counts:
            raise ValueError(f"Duplicate archive run_id: {run_id!r}")
        counts[run_id] = {"traces": 0, "logs": 0}
        for signal in ("traces", "logs"):
            with (directory / f"{signal}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        data, count = _encode_archive_batch(json.loads(line), signal, run_id)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"Invalid {signal} batch in {directory}") from error
                    if count:
                        batches.append((signal, data))
                        counts[run_id][signal] += count
    if not counts:
        raise ValueError("Select at least one archive directory")

    import requests
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceResponse
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

    response_types = {"traces": ExportTraceServiceResponse, "logs": ExportLogsServiceResponse}
    with requests.Session() as session:
        session.trust_env = False
        for index, (signal, data) in enumerate(batches):
            try:
                response = session.post(
                    f"{base_url}/v1/{signal}", data=data,
                    headers={"Authorization": token, "Content-Type": "application/x-protobuf"},
                    timeout=timeout, allow_redirects=False,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                result = response_types[signal].FromString(response.content)
                partial = result.partial_success
                rejected = partial.rejected_spans if signal == "traces" else partial.rejected_log_records
                if rejected or partial.error_message:
                    raise RuntimeError(f"Partial OTLP acceptance ({rejected} rejected)")
            except Exception as error:
                raise RuntimeError(
                    f"Logfire {signal} upload failed after {index} accepted batches; "
                    "inspect the backend before retrying"
                ) from error
    return counts


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
