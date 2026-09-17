"""Check ``observability.py::configure`` and ``observability.py::span`` without inference.

Run in environments with and without the extra:
``PYTHONBREAKPOINT=0 python -m unit_test.agents.test_observability_prototype``.
Enabled checks use the real SDK and capture console text only, not an OTLP archive.
"""

import sys
from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO

from lits import observability as tracing


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


def main():
    """Check disabled operation first, then the installed or missing dependency path."""
    check_disabled()
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


if __name__ == "__main__":
    main()
