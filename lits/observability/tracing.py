"""Optional LiTS tracing, disabled until explicitly configured.

Console rendering is not a persistent trace archive. Enabled defaults produce
console output only; hosted export requires an explicit opt-in. Optional archives
contain SDK-processed telemetry, including its redaction. This module owns its SDK
instance and explicitly registered model instrumentation; tool instrumentation is separate.

Example::

    from lits import observability as tracing

    tracing.configure(enabled=True, console=True)
    with tracing.span("search.iteration", iteration=1) as current:
        current.set_attribute("selected_id", "n1")
    tracing.configure(enabled=False)

To retain telemetry, use a new directory for each run::

    tracing.configure(enabled=True, console=False,
                      archive_dir="runs/example-1", run_id="example-1")
    try:
        with tracing.span("search.iteration", iteration=1):
            pass
        flushed = tracing.flush()
    finally:
        complete = tracing.close()  # After all span contexts have exited.

The directory contains:

=============== ================ ================== ==========================================
File            Format           Purpose            Possible content
=============== ================ ================== ==========================================
traces.jsonl    OTLP JSON Lines  Saved spans        Names, trace/span/parent IDs, timestamps,
                                                    resource and span attributes.
logs.jsonl      OTLP JSON Lines  Saved log records  Timestamp, severity, body, attributes,
                                                    associated trace/span IDs when available.
archive.json    Custom LiTS JSON Archive status     run_id; status: open/complete/incomplete;
                                                    error message when a failure is recorded.
=============== ================ ================== ==========================================

``archive.json`` is separate metadata, not an OTLP record; exclude it from OTLP imports.
A completed archive means normal exporter shutdown, not task success or complete instrumentation.
An interrupted process leaves an open archive; no automatic resume is attempted.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .archive import _Archive

if TYPE_CHECKING:
    from logfire import Logfire, LogfireSpan

_logfire_instance: Logfire | None = None
_archive: _Archive | None = None
_instrumentations = ExitStack()
_logging_handlers: dict[logging.Logger, logging.Handler] = {}


class _NoOpSpan:
    """Accept attributes without storing or exporting them when tracing is off."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Discard an attribute; keep the same call shape as an enabled span."""


def _get_logfire() -> Logfire | None:
    """Return the in-process SDK instance, not a UI backend; None when disabled."""
    return _logfire_instance


def _validate_export_environment() -> None:
    """Reject implicit export destinations before importing or initializing the SDK.

    OTLP endpoint overrides, including loopback addresses, are unsupported: a
    hosted-export opt-in does not authorize a separate OTLP destination. Detector
    plugins are also rejected because they can perform network I/O. Report names,
    not values, since endpoint URLs can contain credentials.

    Model settings such as OPENAI_API_BASE are unrelated and remain untouched.
    """
    for name, value in os.environ.items():
        if value and (
            (name.startswith("OTEL_EXPORTER_OTLP") and name.endswith("_ENDPOINT"))
            or name == "OTEL_EXPERIMENTAL_RESOURCE_DETECTORS"
        ):
            raise ValueError(f"LiTS tracing does not accept {name}; unset it before enabling")


def configure(
    *, enabled: bool = False, send_to_logfire: bool = False, console: bool = True,
    archive_dir: str | os.PathLike[str] | None = None, run_id: str | None = None,
) -> None:
    """Enable, replace, or disable LiTS tracing between runs, outside active spans.

    ========================================  =====================================
    Configuration                             Meaning
    ========================================  =====================================
    enabled=False                             No SDK instance; no LiTS telemetry.
    enabled=True, send_to_logfire=False        Local telemetry; no Logfire upload.
    enabled=True, send_to_logfire=True         Enable export to Logfire's service.
    ========================================  =====================================

    The SDK instance lives in Python, not in a UI server. Hosted export requires
    credentials/configuration; the flag does not open a browser UI. Console output
    is independent of local archive writing and hosted export.

    Args:
        enabled: Import the optional SDK only when True. False shuts down an
            existing LiTS SDK instance and restores no-op spans.
        send_to_logfire: Explicit opt-in to hosted export; credentials alone do
            not enable it. False keeps default telemetry local.
        console: Render spans in the command-line terminal, independently of
            hosted export. This does not create a file or browser UI.
        archive_dir: New per-run directory for OTLP traces/logs and archive status.
            Requires enabled=True and run_id; existing directories are rejected.
        run_id: Non-empty run identity, saved as the lits.run_id resource attribute
            on both traces and logs. Required only when archive_dir is supplied.

    OTLP endpoint and OpenTelemetry detector overrides are rejected before SDK
    initialization, even with hosted export enabled. Custom processors and SDK
    option passthrough are unsupported. Logfire detectors are disabled; explicit
    booleans override environment/file defaults, so stored credentials cannot
    authorize upload. Preflight rejection leaves an existing instance unchanged.
    Other libraries' telemetry configuration is not managed by this facade.
    """
    global _logfire_instance, _archive
    if any(type(flag) is not bool for flag in (enabled, send_to_logfire, console)):
        raise TypeError("enabled, send_to_logfire, and console must be booleans")
    if archive_dir is None and run_id is not None:
        raise ValueError("run_id requires archive_dir")
    if archive_dir is not None and (
        not enabled or not isinstance(run_id, str) or not run_id.strip()
    ):
        raise ValueError("archive_dir requires enabled=True and a non-empty run_id")

    if enabled:
        # Logfire can enable OTLP export independently of send_to_logfire.
        _validate_export_environment()
        try:
            import logfire
        except ModuleNotFoundError as error:
            if error.name != "logfire":
                raise
            raise ImportError("Enable tracing with: pip install 'lits-llm[observability]'") from error

    archive = _Archive(Path(archive_dir), run_id) if archive_dir is not None else None
    close()
    if not enabled:
        return

    try:
        span_processors, log_processors = [], []
        archive_options = {}
        if archive is not None:
            from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.sampling import ALWAYS_ON

            span_processors = [SimpleSpanProcessor(archive.span_exporter)]
            log_processors = [SimpleLogRecordProcessor(archive.log_exporter)]
            archive_options = dict(
                resource_attributes={"lits.run_id": run_id},
                sampling=logfire.SamplingOptions(head=ALWAYS_ON), min_level=0,
            )
        _logfire_instance = logfire.configure(
            local=True,  # Isolate SDK configuration; this flag does not mean offline.
            send_to_logfire=send_to_logfire,
            console=logfire.ConsoleOptions() if console else False,
            service_name="lits",
            metrics=False,
            inspect_arguments=False,
            additional_span_processors=span_processors,
            advanced=logfire.AdvancedOptions(
                resource_detectors=[], log_record_processors=log_processors,
            ),
            **archive_options,
        )
    except Exception as error:
        if archive is not None:
            archive.fail(f"SDK initialization failed: {error}")
            archive.close_exporters()
            archive.write_status("incomplete")
        raise
    _archive = archive


def flush() -> bool:
    """Flush ended spans/logs without closing the run; return False after any failure.

    Open span contexts are not yet exportable. A successful flush leaves the
    archive open; only close() can mark normal completion. Export errors are
    reported without raising into search execution.
    """
    if _logfire_instance is None:
        return True
    try:
        successful = _logfire_instance.force_flush()
    except Exception as error:
        successful = False
        if _archive is not None:
            _archive.fail(f"Telemetry flush failed: {error}")
    if _archive is not None:
        if not successful:
            _archive.fail("Telemetry flush failed")
        return _archive.write_status("open")
    return successful


def close() -> bool:
    """Flush and close the current SDK/files after all span contexts have exited.

    Return whether this shutdown completed without known export failures. With
    no active instance, return True. Consult archive.json for a closed run's
    persisted status; completion describes telemetry writing, not search success.
    Remove client instrumentation and logging handlers installed by this facade
    before SDK shutdown; other file/console handlers remain attached.
    """
    global _logfire_instance, _archive
    for logger, handler in _logging_handlers.items():
        logger.removeHandler(handler)
        handler.close()
    _logging_handlers.clear()
    successful = flush()
    try:
        _instrumentations.close()
        if _logfire_instance is not None:
            successful = _logfire_instance.shutdown() and successful
    except Exception as error:
        successful = False
        if _archive is not None:
            _archive.fail(f"Telemetry shutdown failed: {error}")
    finally:
        _logfire_instance = None
        archive, _archive = _archive, None
    if archive is not None:
        archive.close_exporters()
        if not successful:
            archive.fail("Telemetry shutdown failed")
        successful = archive.write_status("complete")
    return successful


def attach_logging_handler(
    logger: logging.Logger, *, level: int = logging.INFO,
    namespace_filter: logging.Filter | None = None,
) -> None:
    """Attach one LiTS-owned Logfire handler to a logger when tracing is enabled.

    Configure tracing before setup_logging(); that function supplies the file
    handler's level and namespace filter. Repeated calls refresh those settings
    and reattach the same handler after a logger reset, without duplicates.
    close()/reconfiguration detaches it; call setup_logging() for the next run.
    Disabled calls do nothing and do not import Logfire.

    Existing logger.info(...) calls become zero-duration log spans under the
    active span, saved in traces.jsonl, not native OTLP entries in logs.jsonl.
    File/console handlers remain independent. When SDK instrumentation is
    suppressed, only those existing handlers write the message.
    """
    logfire_instance = _get_logfire()
    if logfire_instance is None:
        return
    from logfire import LogfireLoggingHandler

    handler = _logging_handlers.get(logger)
    if handler is None:
        handler = LogfireLoggingHandler(
            logfire_instance=logfire_instance, fallback=logging.NullHandler(),
        )
        _logging_handlers[logger] = handler
    handler.setLevel(level)
    handler.filters.clear()
    if namespace_filter is not None:
        handler.addFilter(namespace_filter)
    logger.addHandler(handler)


def instrument_openai(client: Any) -> None:
    """Instrument one existing OpenAI client when tracing is enabled.

    Configure tracing before constructing model clients. The SDK ignores repeated
    registration of the same client. Retain both its request/response attributes
    and Gen AI fields for inspecting original messages and rendering LLM panels.
    SDK scrubbing still applies; custom extra_body is recorded by the caller.

    close() or reconfiguration removes instrumentation installed by this facade.
    Reused clients must be registered again after enabling a new configuration.
    Disabled calls neither modify the client nor import the optional SDK.
    """
    logfire_instance = _get_logfire()
    if logfire_instance is not None:
        _instrumentations.enter_context(logfire_instance.instrument_openai(client, version=(1, 2)))


def span(name: str, **attributes: Any) -> AbstractContextManager[LogfireSpan | _NoOpSpan]:
    """Return a named span context, or a dependency-free no-op context when disabled.
    1. This avoids:
    ```
    if tracing_enabled:
        with span(...):
            ...
    else:
        ...
    ```

    2. Attributes are passed to Logfire unchanged, allowing
    ```
    with tracing.span(
        "search.iteration",
        iteration=3,
        frontier_size=12,
        budget=5.4,
    ):
        ...
    ```

    3. Both modes support
    ``current.set_attribute(key, value)``.

    4. Both modes propagate exceptions from the body. Enabled nested contexts establish parent/child spans automatically.
    例如：

        with tracing.span("agent.run"):

            with tracing.span("search.iteration"):

                with tracing.span("llm.call"):
                    call_llm()

    开启 tracing 时, Logfire/OpenTelemetry 会自动建立：

        agent.run
        └── search.iteration
            └── llm.call

    而不是三个彼此无关的 spans。

    """
    logfire_instance = _get_logfire()
    if logfire_instance is None:
        return nullcontext(_NoOpSpan())
    return logfire_instance.span(name, **attributes)


def observe(kind: str, record: dict[str, Any], settings: dict[str, dict[str, Any]]) -> None:
    """Emit a named snapshot of selected fields under the active span.

    Args:
        kind: Object kind used to look up a rule in settings.
        record: Source data; selected values are deep-copied before logging.
        settings: Rules mapping kinds to a display name and field names, e.g.::

            {"node": {"name": "search.node", "fields": ("id", "state", "score")}}

    Disabled tracing or an unconfigured kind returns without reading record.
    A configured field missing from record raises KeyError. Rules affect only
    these explicit snapshots, not independently instrumented operations.

    SDK scrubbing and attribute limits apply to the copy, not source data.
    Logfire info records are zero-duration spans, saved in traces.jsonl.
    With archive saving enabled, the export pipeline is::

        observe(...)
          -> Logfire.info(...)
          -> zero-duration span
          -> SimpleSpanProcessor
          -> span_exporter
          -> traces.jsonl

    This does not emit native OpenTelemetry LogRecords through log_exporter.
    Nesting follows the active span, not dots in the configured display name.
    """
    logfire_instance = _get_logfire()
    if logfire_instance is None:
        return
    rule = settings.get(kind)
    if rule is None:
        return
    displayed = {field: deepcopy(record[field]) for field in rule["fields"]}
    logfire_instance.info(rule["name"], **displayed)
