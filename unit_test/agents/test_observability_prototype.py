"""Check ``observability.py::configure`` and ``observability.py::span`` without inference.

Run in environments with and without the extra:
``PYTHONBREAKPOINT=0 python -m unit_test.agents.test_observability_prototype``.
Enabled checks use the real SDK and capture console text only, not an OTLP archive.
The standalone process denies network access, including background SDK requests.
"""

import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from lits import observability as tracing


@contextmanager
def telemetry_environment(**settings):
    """Isolate telemetry settings and restore the caller's environment afterward."""
    previous = dict(os.environ)
    try:
        for name in tuple(os.environ):
            if name.startswith(("LOGFIRE_", "OTEL_")):
                del os.environ[name]
        os.environ.update(settings)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def check_rejected_configuration():
    """Return the case count after checking rejection without SDK initialization/replacement."""
    original = tracing._get_logfire()
    sdk_loaded = "logfire" in sys.modules
    endpoint_names = (
        "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    )
    values = ("https://secret@telemetry.invalid:4318", "http://127.0.0.1:4318")
    cases = [({name: value}, {}) for name in endpoint_names for value in values]
    cases += [
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": values[0]}, {"send_to_logfire": True}),
        ({"OTEL_EXPERIMENTAL_RESOURCE_DETECTORS": "process"}, {}),
        ({}, {"send_to_logfire": None}),
        ({}, {"send_to_logfire": "if-token-present"}),
        ({}, {"additional_span_processors": []}),
        ({}, {"advanced": {}}),
        ({}, {"metrics": {}}),
    ]
    for environment, options in cases:
        with telemetry_environment(**environment):
            try:
                tracing.configure(enabled=True, **options)
            except (TypeError, ValueError) as error:
                if "secret@" in str(error):
                    raise RuntimeError("Validation exposed endpoint credentials") from error
            else:
                raise RuntimeError("Unsupported configuration was accepted")
            if tracing._get_logfire() is not original or ("logfire" in sys.modules) != sdk_loaded:
                raise RuntimeError("Rejected configuration changed the SDK instance/import state")
            if original is None:
                tracing.configure(enabled=False)  # Disabling needs no endpoint validation.
    return len(cases)


def check_exception_propagation():
    """Keep the application's original exception in either tracing mode."""
    failure = RuntimeError("application failure")
    try:
        with tracing.span("exception.check"):
            raise failure
    except RuntimeError as error:
        if error is not failure:
            raise RuntimeError("Tracing replaced the application exception") from error
    else:
        raise RuntimeError("Tracing swallowed the application exception")


def check_disabled():
    """Use the default no-op and explicit disable without importing the optional SDK."""
    for explicit in (False, True):
        if explicit:
            tracing.configure(enabled=False)
        with tracing.span("disabled.iteration", iteration=1) as current:
            current.set_attribute("selected_id", "n1")
        check_exception_propagation()
    if tracing._get_logfire() is not None or "logfire" in sys.modules:
        raise RuntimeError("Disabled tracing initialized or imported the SDK")
    print("PASS default/explicit disabled spans and exception propagation; no SDK import")
    breakpoint()  # inspect: tracing._get_logfire(), "logfire" in sys.modules


def check_enabled():
    """Use real local console spans; reconfiguration must not duplicate output."""
    from opentelemetry.trace import get_tracer_provider

    original_provider = get_tracer_provider()
    output = StringIO()
    try:
        with redirect_stdout(output):
            for iteration in range(2):
                tracing.configure(enabled=True)
                logfire_instance = tracing._get_logfire()
                if logfire_instance.config.send_to_logfire is not False:
                    raise RuntimeError("Hosted export must be disabled by default")
                with tracing.span(f"visible.iteration.{iteration}", iteration=iteration):
                    with tracing.span(f"visible.node.{iteration}") as current:
                        current.set_attribute("selected_id", "n1")
            visible = output.getvalue()
            for iteration in range(2):
                for kind in ("iteration", "node"):
                    if visible.count(f"visible.{kind}.{iteration}") != 1:
                        raise RuntimeError(f"Missing or duplicated span output: {visible!r}")

            tracing.configure(enabled=True, console=False)
            check_rejected_configuration()
            with tracing.span("hidden.iteration") as current:
                current.set_attribute("selected_id", "n2")
            check_exception_propagation()
            tracing.configure(enabled=False)
            with tracing.span("disabled.again"):
                pass
            if output.getvalue() != visible or tracing._get_logfire() is not None:
                raise RuntimeError("Quiet or disabled tracing emitted output or stayed enabled")
        if get_tracer_provider() is not original_provider:
            raise RuntimeError("LiTS replaced the process-global tracer provider")
    finally:
        tracing.configure(enabled=False)
    print("PASS real SDK: local defaults, nested spans, reconfiguration, console off, shutdown")
    print("PASS enabled exception propagation and unchanged global tracer provider")
    breakpoint()  # inspect: visible, tracing._get_logfire()


def check_stored_configuration():
    """Stored credentials and online defaults cannot override explicit local-only setup."""
    with TemporaryDirectory(prefix="lits-telemetry-config-") as directory:
        root = Path(directory)
        (root / "pyproject.toml").write_text(
            '[tool.logfire]\nsend_to_logfire = true\n', encoding="utf-8",
        )
        (root / "logfire_credentials.json").write_text(json.dumps({
            "token": "not-a-real-token", "project_name": "offline-check",
            "project_url": "https://telemetry.invalid/project",
            "logfire_api_url": "https://telemetry.invalid",
        }), encoding="utf-8")
        for overrides in ({}, {"LOGFIRE_SEND_TO_LOGFIRE": "true", "LOGFIRE_TOKEN": "unused"}):
            with telemetry_environment(
                LOGFIRE_CONFIG_DIR=directory, LOGFIRE_CREDENTIALS_DIR=directory,
                LOGFIRE_RESOURCE_DETECTORS="*", **overrides,
            ):
                try:
                    tracing.configure(enabled=True, console=False)
                    config = tracing._get_logfire().config
                    if config.send_to_logfire is not False or config.advanced.resource_detectors:
                        raise RuntimeError("Stored settings enabled export or detector plugins")
                    if config.data_dir != root:
                        raise RuntimeError("Check did not use the temporary credentials directory")
                    with tracing.span("stored.configuration"):
                        pass
                finally:
                    tracing.configure(enabled=False)
    print("PASS file/environment online defaults and stored credentials remain local-only")


def check_model_endpoint():
    """Construct the approved Tailscale client unchanged; do not call the model."""
    from lits import get_lm

    endpoint = "https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1"
    with telemetry_environment(OPENAI_API_BASE=endpoint):
        try:
            tracing.configure(enabled=True, console=False)
            model = get_lm("openai/local-instruct", api_key="no-key")
            try:
                if str(model.client.base_url).rstrip("/") != endpoint:
                    raise RuntimeError("Telemetry configuration changed the model endpoint")
            finally:
                model.client.close()
        finally:
            tracing.configure(enabled=False)
    print("PASS Tailscale model-client construction; no inference requested")


def main():
    """Check disabled operation first, then the installed or missing dependency path."""
    network_attempts = []

    def deny_network(event, args):
        """Block and record socket activity without replacing the SDK or its exporters."""
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            network_attempts.append(event)
            raise PermissionError("Network access is forbidden in this offline check")

    sys.addaudithook(deny_network)
    with telemetry_environment():
        check_disabled()
        count = check_rejected_configuration()
        print(f"PASS {count} rejected configurations before SDK initialization")
        if find_spec("logfire") is None:
            try:
                tracing.configure(enabled=True)
            except ImportError as error:
                if "lits-llm[observability]" not in str(error):
                    raise RuntimeError("Missing-extra error lacks an installation hint") from error
            else:
                raise RuntimeError("Enabling without Logfire should fail explicitly")
            print("PASS missing-extra installation hint; enabled SDK checks skipped")
        else:
            check_enabled()
            check_stored_configuration()
            check_model_endpoint()
    if network_attempts:
        raise RuntimeError(f"Unexpected network attempts: {network_attempts}")
    print("PASS offline checks: zero socket/DNS attempts")


if __name__ == "__main__":
    main()
