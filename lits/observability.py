"""Optional LiTS tracing, disabled until explicitly configured.

Console rendering is not a persistent trace archive. Enabled defaults produce
console output only; hosted export requires an explicit opt-in. This module does
not save traces or install model/tool instrumentation, and owns its SDK instance.

Example::

    from lits import observability as tracing

    tracing.configure(enabled=True, console=True)
    with tracing.span("search.iteration", iteration=1) as current:
        current.set_attribute("selected_id", "n1")
    tracing.configure(enabled=False)
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logfire import Logfire, LogfireSpan

_logfire_instance: Logfire | None = None


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
    is independent. Local file export is not implemented here yet.

    Args:
        enabled: Import the optional SDK only when True. False shuts down an
            existing LiTS SDK instance and restores no-op spans.
        send_to_logfire: Explicit opt-in to hosted export; credentials alone do
            not enable it. False keeps default telemetry local.
        console: Render spans in the command-line terminal, independently of
            hosted export. This does not create a file or browser UI.

    OTLP endpoint and OpenTelemetry detector overrides are rejected before SDK
    initialization, even with hosted export enabled. Custom processors and SDK
    option passthrough are unsupported. Logfire detectors are disabled; explicit
    booleans override environment/file defaults, so stored credentials cannot
    authorize upload. Preflight rejection leaves an existing instance unchanged.
    Other libraries' telemetry configuration is not managed by this facade.
    """
    global _logfire_instance
    if any(type(flag) is not bool for flag in (enabled, send_to_logfire, console)):
        raise TypeError("enabled, send_to_logfire, and console must be booleans")

    if enabled:
        # Logfire can enable OTLP export independently of send_to_logfire.
        _validate_export_environment()
        try:
            import logfire
        except ModuleNotFoundError as error:
            if error.name != "logfire":
                raise
            raise ImportError("Enable tracing with: pip install 'lits-llm[observability]'") from error

    if _logfire_instance is not None:
        _logfire_instance.shutdown()
        _logfire_instance = None
    if enabled:
        _logfire_instance = logfire.configure(
            local=True,  # Isolate SDK configuration; this flag does not mean offline.
            send_to_logfire=send_to_logfire,
            console=logfire.ConsoleOptions() if console else False,
            service_name="lits",
            metrics=False,
            inspect_arguments=False,
            advanced=logfire.AdvancedOptions(resource_detectors=[]),
        )


def span(name: str, **attributes: Any) -> AbstractContextManager[LogfireSpan | _NoOpSpan]:
    """Return a named span context, or a dependency-free no-op context when disabled.

    Attributes are passed to Logfire unchanged. Both modes support
    ``current.set_attribute(key, value)`` and propagate exceptions from the body.
    Enabled nested contexts establish parent/child spans automatically.
    """
    logfire_instance = _get_logfire()
    if logfire_instance is None:
        return nullcontext(_NoOpSpan())
    return logfire_instance.span(name, **attributes)
