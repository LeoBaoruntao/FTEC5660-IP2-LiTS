"""Check ``observability/tracing.py::configure`` and ``observability/tracing.py::span`` without inference.

Run in environments with and without the extra:
``PYTHONBREAKPOINT=0 python -m unit_test.observability.test_observability_prototype``.
Enabled checks use the real SDK. Archive checks retain files in a new temporary
directory and parse them using OpenTelemetry's OTLP JSON message classes.
The default offline mode denies network access, including background SDK requests.

The model-backed end-to-end BFS test is a separate script:
``python -m unit_test.observability.test_bfs_observability``.
"""

import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from lits import observability as tracing
from lits.observability import tracing as tracing_impl


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
    original = tracing_impl._get_logfire()
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
            if tracing_impl._get_logfire() is not original or ("logfire" in sys.modules) != sdk_loaded:
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
            tracing.observe("node", {}, {
                "node": {"name": "search.node", "fields": ("state",)},
            })  # Disabled observation must not read the missing field or import Logfire.
        check_exception_propagation()
    if tracing_impl._get_logfire() is not None or "logfire" in sys.modules:
        raise RuntimeError("Disabled tracing initialized or imported the SDK")
    print("PASS default/explicit disabled spans and exception propagation; no SDK import")
    breakpoint()  # inspect: tracing_impl._get_logfire(), "logfire" in sys.modules


def check_enabled():
    """Use real local console spans; reconfiguration must not duplicate output."""
    from opentelemetry.trace import get_tracer_provider

    original_provider = get_tracer_provider()
    output = StringIO()
    try:
        with redirect_stdout(output):
            for iteration in range(2):
                tracing.configure(enabled=True)
                logfire_instance = tracing_impl._get_logfire()
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
            if output.getvalue() != visible or tracing_impl._get_logfire() is not None:
                raise RuntimeError("Quiet or disabled tracing emitted output or stayed enabled")
        if get_tracer_provider() is not original_provider:
            raise RuntimeError("LiTS replaced the process-global tracer provider")
    finally:
        tracing.configure(enabled=False)
    print("PASS real SDK: local defaults, nested spans, reconfiguration, console off, shutdown")
    print("PASS enabled exception propagation and unchanged global tracer provider")
    breakpoint()  # inspect: visible, tracing_impl._get_logfire()


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
                    config = tracing_impl._get_logfire().config
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
                instrumented_request = model.client.request
                tracing.instrument_openai(model.client)
                if model.client.request is not instrumented_request:
                    raise RuntimeError("Repeated instrumentation changed the request wrapper")
                tracing.configure(enabled=False)
                plain_request = model.client.request
                if plain_request is instrumented_request:
                    raise RuntimeError("Disabling tracing did not remove client instrumentation")
                tracing.instrument_openai(model.client)
                if model.client.request is not plain_request:
                    raise RuntimeError("Disabled registration changed the client")
                tracing.configure(enabled=True, console=False)
                tracing.instrument_openai(model.client)
                if model.client.request is plain_request:
                    raise RuntimeError("Client was not registered for the new configuration")
            finally:
                model.client.close()
        finally:
            tracing.configure(enabled=False)
    print("PASS Tailscale client, duplicate registration, cleanup and reconfiguration; no inference")


def read_archive(directory, run_id):
    """Round-trip OTLP messages, verify run resources, and return span/log records."""
    from opentelemetry.proto_json.collector.logs.v1.logs_service import ExportLogsServiceRequest
    from opentelemetry.proto_json.collector.trace.v1.trace_service import ExportTraceServiceRequest

    records = {}
    for filename, message_type, resources_key, scopes_key, records_key in (
        ("traces.jsonl", ExportTraceServiceRequest, "resourceSpans", "scopeSpans", "spans"),
        ("logs.jsonl", ExportLogsServiceRequest, "resourceLogs", "scopeLogs", "logRecords"),
    ):
        records[records_key] = []
        for line in (directory / filename).read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if message_type.from_dict(payload).to_dict() != payload:
                raise RuntimeError("OTLP JSON round-trip changed the archive")
            for resource in payload[resources_key]:
                attrs = {entry["key"]: entry["value"] for entry in resource["resource"]["attributes"]}
                if (attrs["lits.run_id"] != {"stringValue": run_id}
                        or attrs["service.name"] != {"stringValue": "lits"}):
                    raise RuntimeError("Archive resource has the wrong run or service identity")
                for scope in resource[scopes_key]:
                    records[records_key].extend(scope[records_key])
    return records


def check_archive_filter(source_root, run_ids, incomplete_run_id):
    """Check archive.py::filter_archives using the real SDK files from check_archives.

    Select two runs, relocate their copies, and compare bytes plus decoded span/log
    records. Reject invalid selections before writing anything; never infer which
    answers were incorrect. No new inference, SDK configuration, or upload occurs.
    """
    output_root = Path(mkdtemp(prefix="lits-selected-archives-")).resolve()
    selected = sorted(run_ids)[:2]
    originals = {path: path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    output = output_root / "selected"
    copied = tracing.filter_archives(source_root, selected, output)
    if copied != [output / run_id for run_id in selected]:
        raise RuntimeError("Archive selection changed the requested order or layout")
    relocated = output_root / "relocated"
    tracing.filter_archives(output, selected, relocated)
    for root in (output, relocated):
        actual = {path.parent.relative_to(root).as_posix() for path in root.rglob("archive.json")}
        if actual != set(selected):
            raise RuntimeError("Selected output included an unselected run or omitted a selected run")
        for run_id in selected:
            for filename in ("traces.jsonl", "logs.jsonl", "archive.json"):
                source = source_root / run_id / filename
                if (root / run_id / filename).read_bytes() != originals[source]:
                    raise RuntimeError("Selection or relocation changed archive bytes")
            expected = read_archive(source_root / run_id, run_id)
            if not expected["logRecords"] or read_archive(root / run_id, run_id) != expected:
                raise RuntimeError("Relocation lost native logs, identity, attributes, or span linkage")

    cases = (
        ([], ValueError), (selected[0], ValueError),
        ([selected[0], selected[0]], ValueError),
        ([str(source_root / selected[0])], ValueError),
        (["../escape"], ValueError), ([selected[0] + "/"], ValueError),
        ([selected[0], "unknown-run"], FileNotFoundError),
        ([selected[0], incomplete_run_id], ValueError),
    )
    for index, (selection, exception) in enumerate(cases):
        destination = output_root / f"rejected-{index}"
        try:
            tracing.filter_archives(source_root, selection, destination)
        except exception:
            if destination.exists():
                raise RuntimeError("Invalid selection created partial output")
        else:
            raise RuntimeError(f"Invalid selection was accepted: {selection}")
    for destination, exception in ((output, FileExistsError), (source_root / "selection", ValueError)):
        try:
            tracing.filter_archives(source_root, selected, destination)
        except exception:
            pass
        else:
            raise RuntimeError("Existing or overlapping output was accepted")
    for path, contents in originals.items():
        if path.read_bytes() != contents:
            raise RuntimeError("Selection modified a source archive")
    if any((output / path.relative_to(source_root)).read_bytes() != contents
           for path, contents in originals.items() if path.parent.relative_to(source_root).as_posix() in selected):
        raise RuntimeError("Rejected overwrite modified the selected output")
    print(f"PASS archive selection: two of six runs, byte-identical relocation, native logs, ten rejections: {output_root}")
    breakpoint()  # inspect: selected, copied, relocated, originals


def check_archives():
    """Check observability/tracing.py::configure with caller-resolved result paths.

    Vary dataset, method, example, repeat, and attempt independently. For example,
    result_dir=results_root/"math500/beam_pointwise/run_0" produces an archive at
    result_dir/"telemetry/example-42-repeat-0-attempt-0"; its ID is that full path
    relative to results_root, not the shared "run_0" or execution basename.

    Check real file export, linked log records, run isolation, and failure status.
    Native OpenTelemetry logs and Logfire info events must use separate files.
    """
    from opentelemetry.sdk._logs import LoggingHandler

    results_root = Path(mkdtemp(prefix="lits-otlp-check-"))
    cases = (
        ("math500/beam_pointwise/run_0", 42, 0, 0),
        ("math500/beam_llm/run_0", 42, 0, 0),
        ("dbbench/beam_pointwise/run_0", 42, 0, 0),
        ("math500/beam_pointwise/run_0", 43, 0, 0),
        ("math500/beam_pointwise/run_0", 42, 1, 0),
        ("math500/beam_pointwise/run_0", 42, 0, 1),
    )
    trace_ids, run_ids, saved_files = set(), set(), {}
    for experiment, query_idx, repetition_index, attempt_index in cases:
        result_dir = results_root / experiment
        result_dir.mkdir(parents=True, exist_ok=True)
        execution = f"example-{query_idx}-repeat-{repetition_index}-attempt-{attempt_index}"
        directory = result_dir / "telemetry" / execution
        run_id = directory.relative_to(results_root).as_posix()
        with telemetry_environment(LOGFIRE_TRACE_SAMPLE_RATE="0", LOGFIRE_MIN_LEVEL="error"):
            tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
            logger = logging.Logger("archive.check")
            handler = LoggingHandler(logger_provider=tracing_impl._get_logfire().config.get_logger_provider())
            logger.addHandler(handler)
            try:
                with tracing.span("search.run"):
                    with tracing.span("search.node", node_id="n1", score=0.75):
                        logger.info("selected n1", extra={"node_id": "n1"})
                        tracing_impl._get_logfire().info("selection.recorded", node_id="n1")
                if not tracing.flush():
                    raise RuntimeError("Healthy archive failed to flush")
                if json.loads((directory / "archive.json").read_text())["status"] != "open":
                    raise RuntimeError("Flush must not mark an active archive complete")
                original = tracing_impl._get_logfire()
                try:
                    tracing.configure(enabled=True, archive_dir=directory, run_id=run_id)
                except FileExistsError:
                    if tracing_impl._get_logfire() is not original:
                        raise RuntimeError("Existing-directory rejection replaced the active instance")
                else:
                    raise RuntimeError("Existing archive was reused")
            finally:
                logger.removeHandler(handler)
                handler.close()
                complete = tracing.close()
            status = json.loads((directory / "archive.json").read_text())
            if not complete or status != {"run_id": run_id, "status": "complete"}:
                raise RuntimeError(f"Incorrect archive completion: {status}")

        records = read_archive(directory, run_id)
        spans = {span["name"]: span for span in records["spans"]}
        if len(records["spans"]) != 3 or set(spans) != {
            "search.run", "search.node", "selection.recorded",
        }:
            raise RuntimeError("Missing or extra archived spans")
        parent, child = spans["search.run"], spans["search.node"]
        if child["parentSpanId"] != parent["spanId"] or child["traceId"] != parent["traceId"]:
            raise RuntimeError("Archive lost span hierarchy")
        if not (0 < int(parent["startTimeUnixNano"]) <= int(child["startTimeUnixNano"])
                <= int(child["endTimeUnixNano"]) <= int(parent["endTimeUnixNano"])):
            raise RuntimeError("Archive lost span timestamps")
        attrs = {entry["key"]: entry["value"] for entry in child["attributes"]}
        if attrs["node_id"] != {"stringValue": "n1"} or attrs["score"] != {"doubleValue": 0.75}:
            raise RuntimeError("Archive changed node attributes")
        event = spans["selection.recorded"]
        event_attrs = {entry["key"]: entry["value"] for entry in event["attributes"]}
        if (event["parentSpanId"], event["traceId"]) != (child["spanId"], child["traceId"]):
            raise RuntimeError("Logfire info event lost its span association")
        if (event_attrs["logfire.span_type"] != {"stringValue": "log"}
                or event_attrs["node_id"] != {"stringValue": "n1"}
                or event["startTimeUnixNano"] != event["endTimeUnixNano"]):
            raise RuntimeError("Logfire info event lost its type, attributes, or zero duration")
        if len(records["logRecords"]) != 1:
            raise RuntimeError("Missing or duplicated forwarded log")
        record = records["logRecords"][0]
        if (record["traceId"], record["spanId"], record["body"]) != (
            child["traceId"], child["spanId"], {"stringValue": "selected n1"},
        ):
            raise RuntimeError("Forwarded log lost its message or span association")
        trace_ids.add(parent["traceId"])
        run_ids.add(run_id)
        for filename in ("traces.jsonl", "logs.jsonl", "archive.json"):
            path = directory / filename
            saved_files[path] = path.read_bytes()
    if len(trace_ids) != len(cases) or len(run_ids) != len(cases):
        raise RuntimeError("Independent executions reused a run or trace identity")

    # Reopening the last completed archive must fail just as reopening an active one did.
    try:
        tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
    except FileExistsError:
        if tracing_impl._get_logfire() is not None:
            raise RuntimeError("Completed-directory rejection initialized the SDK")
    else:
        tracing.close()
        raise RuntimeError("Completed archive was reused")

    for attempt_index, failed_pipeline in enumerate(("span", "log"), start=2):
        directory = (results_root / cases[0][0] / "telemetry"
                     / f"example-42-repeat-0-attempt-{attempt_index}")
        run_id = directory.relative_to(results_root).as_posix()
        tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
        logger = logging.Logger("archive.failure.check")
        handler = LoggingHandler(logger_provider=tracing_impl._get_logfire().config.get_logger_provider())
        logger.addHandler(handler)
        try:
            archive = tracing_impl._archive
            exporter = archive.span_exporter if failed_pipeline == "span" else archive.log_exporter
            exporter.shutdown()  # Close the real file to force an export failure.
            with tracing.span("closed.exporter"):
                logger.info("export failure must not interrupt execution")
                result = "search result unchanged"
            if tracing.flush() or result != "search result unchanged":
                raise RuntimeError("Export failure was hidden or changed application execution")
        finally:
            logger.removeHandler(handler)
            handler.close()
            complete = tracing.close()
        status = json.loads((directory / "archive.json").read_text())
        if (complete or status["status"] != "incomplete" or not status.get("error")
                or status["run_id"] != run_id):
            raise RuntimeError("Failed archive lost its identity or was marked complete")

    if any(path.read_bytes() != content for path, content in saved_files.items()):
        raise RuntimeError("A later execution or rejected reopen changed an earlier archive")
    print(f"PASS {len(cases)} result-path identities, OTLP round-trips, linked logs/info events")
    print(f"PASS unchanged archives, open/closed overwrite rejection, both exporter failures: {results_root}")
    check_archive_filter(results_root, run_ids, run_id)
    breakpoint()  # inspect: results_root, run_ids, status, records


def check_observations():
    """Check observability/tracing.py::observe with real SDK scrubbing and attribute limits."""
    root = Path(mkdtemp(prefix="lits-observation-check-"))
    settings = {"node": {
        "name": "search.node",
        "fields": ("id", "state", "score", "api_key", "text"),
    }}
    for limit in (None, 256):
        run_id = "unlimited" if limit is None else "limited"
        directory = root / run_id
        record = {
            "id": "n1", "state": {"steps": ["x = 2"]}, "score": 0.75,
            "api_key": "fake-observation-value", "text": "x" * 1024,
            "omitted": "unselected field",
        }
        original, original_settings = deepcopy(record), deepcopy(settings)
        limits = {} if limit is None else {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": str(limit)}
        with telemetry_environment(**limits):
            tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
            try:
                with tracing.span("search.iteration", iteration=1):
                    tracing.observe("node", record, settings)
                    tracing.observe("frontier", {}, settings)  # No rule: no event or field lookup.
                    with tracing.span("independent.operation"):
                        tracing.observe("node", {}, {})  # Hidden snapshots must not suppress spans.
                    if record != original or settings != original_settings:
                        raise RuntimeError("Observation changed source data or settings")
                    record["state"]["steps"].append("x = 3")
                    record["score"] = 1.0
            finally:
                complete = tracing.close()
            if not complete:
                raise RuntimeError("Observation archive did not close successfully")

        records = read_archive(directory, run_id)
        spans = {item["name"]: item for item in records["spans"]}
        if len(records["spans"]) != 3 or set(spans) != {
            "search.iteration", "search.node", "independent.operation",
        }:
            raise RuntimeError("Observation rules hid independent spans or emitted unwanted events")
        parent, snapshot = spans["search.iteration"], spans["search.node"]
        if (snapshot["parentSpanId"], snapshot["traceId"]) != (parent["spanId"], parent["traceId"]):
            raise RuntimeError("Observation is not nested under the active iteration")
        attrs = {item["key"]: item["value"] for item in snapshot["attributes"]}
        if attrs["logfire.span_type"] != {"stringValue": "log"} or records["logRecords"]:
            raise RuntimeError("Logfire observation did not use the span export pipeline")
        if snapshot["startTimeUnixNano"] != snapshot["endTimeUnixNano"]:
            raise RuntimeError("Observation unexpectedly has a duration")
        if (json.loads(attrs["state"]["stringValue"]) != original["state"]
                or attrs["score"] != {"doubleValue": 0.75} or "omitted" in attrs):
            raise RuntimeError("Snapshot changed after capture or included an unselected field")
        if "[Scrubbed" not in attrs["api_key"]["stringValue"]:
            raise RuntimeError("SDK did not redact the selected sensitive field")
        if original["api_key"] in (directory / "traces.jsonl").read_text():
            raise RuntimeError("Sensitive observation value leaked into the archive")
        expected_text = original["text"] if limit is None else original["text"][:limit]
        if attrs["text"] != {"stringValue": expected_text}:
            raise RuntimeError("Archived text does not reflect the configured SDK length limit")
    print(f"PASS observations: nesting, field selection, snapshots, scrubbing, and SDK limits: {root}")
    breakpoint()  # inspect: root, snapshot, attrs, record, original


def check_logging(enabled=False):
    """Check log.py::setup_logging and observability/tracing.py::attach_logging_handler.

    Use real root/child loggers and file/console handlers. Repeat setup at INFO
    and DEBUG, then reconfigure tracing and narrow the namespace filter. Verify
    once-only output, span association, run isolation, and disabled SDK-free use.

    File and telemetry filtering (the console stays at INFO+):

    ==================== ========= =============== ============================================
    Parent span          Threshold Logger prefixes Result
    ==================== ========= =============== ============================================
    logging.info         INFO+     lits, mem0      Keep INFO/WARNING; filter DEBUG.
    logging.debug        DEBUG+    lits, mem0      Keep all three messages.
    logging.reconfigured INFO+     lits            Filter DEBUG by level and memory by prefix.
    ==================== ========= =============== ============================================

    verbose=True sets the file and telemetry threshold to DEBUG. The "memory"
    message is a WARNING from mem0.logging_check, not an actual memory operation.
    """
    from lits.log import setup_logging

    root = Path(mkdtemp(prefix="lits-logging-check-"))
    logger = logging.getLogger()
    previous = logger.handlers[:], logger.level, logger.propagate
    handlers, expected_file, expected_console, expected_spans = set(), [], [], {}
    directories, archived = [], {}
    console = StringIO()
    sdk_loaded = "logfire" in sys.modules
    try:
        with telemetry_environment(LITS_LOG_LEVEL="INFO"), redirect_stderr(console):
            for phase, verbose in (("info", False), ("debug", True), ("reconfigured", False)):
                if phase != "debug":
                    directory = root / phase
                    options = dict(archive_dir=directory, run_id=phase) if enabled else {}
                    tracing.configure(enabled=enabled, console=False, **options)
                    if phase == "reconfigured" and len(logger.handlers) != 2:
                        raise RuntimeError("Reconfiguration did not retain only file/console handlers")
                    if enabled:
                        directories.append((directory, phase))
                        if phase == "reconfigured":
                            archived = {path: path.read_bytes() for path in directories[0][0].iterdir()}
                namespaces = ("lits",) if phase == "reconfigured" else ("lits", "mem0")
                setup_logging("execution", str(root), verbose=verbose, override=False,
                              add_console_handler=True, allowed_namespaces=namespaces, json_format=True)
                handlers.update(logger.handlers)
                attached = logger.handlers[:]
                file_handler = next(item for item in attached if isinstance(item, logging.FileHandler))
                for _ in range(2):
                    tracing.attach_logging_handler(
                        logger, level=file_handler.level, namespace_filter=file_handler.filters[0],
                    )
                if logger.handlers != attached or len(attached) != (3 if enabled else 2):
                    raise RuntimeError("Logging setup duplicated or omitted a handler")
                messages = [f"accepted {phase}"]
                if verbose:
                    messages.append(f"debug {phase}")
                if "mem0" in namespaces:
                    messages.append(f"memory {phase}")
                expected_file.extend(messages)
                expected_console.extend(message for message in messages if not message.startswith("debug"))
                expected_spans[f"logging.{phase}"] = messages
                with tracing.span(f"logging.{phase}"):
                    logging.getLogger("lits.logging_check").info("accepted %s", phase, extra={"node_id": "n1"})
                    logging.getLogger("lits.logging_check").debug("debug %s", phase)
                    logging.getLogger("mem0.logging_check").warning("memory %s", phase)
                    logging.getLogger("outside.logging_check").warning("excluded %s", phase)
                if enabled and phase == "debug":
                    from logfire import suppress_instrumentation

                    with suppress_instrumentation():
                        logging.getLogger("lits.logging_check").info("local suppressed")
                    expected_file.append("local suppressed")
                    expected_console.append("local suppressed")
            tracing.configure(enabled=False)
            if len(logger.handlers) != 2:
                raise RuntimeError("Disabling tracing did not remove its handler")
            logging.getLogger("lits.logging_check").info("after close")
            expected_file.append("after close")
            expected_console.append("after close")
    finally:
        tracing.close()
        for handler in handlers:
            handler.close()
        logger.handlers, logger.level, logger.propagate = previous

    messages = [json.loads(line)["msg"] for line in (root / "execution.log").read_text().splitlines()]
    if messages != expected_file:
        raise RuntimeError(f"Unexpected file output: {messages}")
    console_lines = console.getvalue().splitlines()
    if (len(console_lines) != len(expected_console)
            or any(sum(line.endswith(message) for line in console_lines) != 1 for message in expected_console)):
        raise RuntimeError(f"Unexpected console output: {console.getvalue()}")
    if not enabled and ("logfire" in sys.modules) != sdk_loaded:
        raise RuntimeError("Disabled logging imported Logfire")
    if not enabled and (root / "info").exists():
        raise RuntimeError("Disabled logging created telemetry files")
    if any(path.read_bytes() != data for path, data in archived.items()):
        raise RuntimeError("A later run wrote to a closed archive")
    for directory, run_id in directories:
        if json.loads((directory / "archive.json").read_text())["status"] != "complete":
            raise RuntimeError("Logging archive did not close successfully")
        records = read_archive(directory, run_id)
        parents = {item["spanId"]: item for item in records["spans"] if item["name"] in expected_spans}
        seen = {item["name"]: [] for item in parents.values()}
        for item in records["spans"]:
            if item["spanId"] in parents:
                continue
            attrs = {entry["key"]: entry["value"] for entry in item["attributes"]}
            parent = parents[item["parentSpanId"]]
            if (attrs["logfire.span_type"] != {"stringValue": "log"}
                    or item["traceId"] != parent["traceId"]):
                raise RuntimeError("Forwarded log is not associated with its active span")
            message = attrs["logfire.msg"]["stringValue"]
            if message.startswith("accepted") and attrs["node_id"] != {"stringValue": "n1"}:
                raise RuntimeError("Structured logging fields were lost")
            seen[parent["name"]].append(message)
        if records["logRecords"] or any(messages != expected_spans[name] for name, messages in seen.items()):
            raise RuntimeError("Forwarded records were duplicated, filtered incorrectly, or sent as native logs")
        expected_parents = {"logging.info", "logging.debug"} if run_id == "info" else {"logging.reconfigured"}
        if set(seen) != expected_parents:
            raise RuntimeError("Missing logging spans")
    print(f"PASS logging enabled={enabled}: filtering, repeated setup, lifecycle, local outputs: {root}")
    breakpoint()  # inspect: root, messages, expected_spans, console.getvalue()


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
        check_logging()
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
            check_archives()
            check_observations()
            check_logging(enabled=True)
    if network_attempts:
        raise RuntimeError(f"Unexpected network attempts: {network_attempts}")
    print("PASS offline checks: zero socket/DNS attempts")


if __name__ == "__main__":
    main()
