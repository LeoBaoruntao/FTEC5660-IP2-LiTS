"""Verify archive.py::upload_archives using existing files, never new inference.

Run with --source-root ORIGINAL --selected-root RELOCATED for offline preflight.
Add --upload --credentials-dir .logfire to explicitly import into xinzhel/lits,
or --verify-only to read back previously uploaded records without resending.
The live check refuses previously uploaded selected traces, compares unselected
records before/after, and retains its read-back report under a new temp directory.
"""

import argparse
import base64
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from lits.observability import upload_archives
from lits.observability.archive import _encode_archive_batch
from unit_test.observability.test_observability_prototype import read_archive


def check_preflight(directories):
    """Reject unsafe destinations and invalid archives before any network request."""
    base_url = "https://logfire-us.pydantic.dev"
    cases = [([], {}), ([directories[0]] * 2, {}),
             (directories, {"base_url": "http://logfire-us.pydantic.dev"}),
             (directories, {"token": ""}), (directories, {"timeout": 0})]
    with TemporaryDirectory(prefix="lits-import-invalid-") as temporary:
        directory = Path(temporary)
        original = {p.name: p.read_bytes() for p in directories[0].iterdir() if p.is_file()}
        for change in ("incomplete", "run_id", "malformed", "trace_id"):
            for name, contents in original.items():
                (directory / name).write_bytes(contents)
            marker = json.loads((directory / "archive.json").read_text())
            if change in ("incomplete", "run_id"):
                marker["status" if change == "incomplete" else "run_id"] = change
                (directory / "archive.json").write_text(json.dumps(marker))
            else:
                path = directory / "logs.jsonl"
                payload = json.loads(path.read_text().splitlines()[0])
                payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["traceId"] = "ff"
                path.write_text("not json\n" if change == "malformed" else json.dumps(payload) + "\n")
            try:
                upload_archives([directories[1], directory], base_url=base_url, token="unused")
            except ValueError:
                pass
            else:
                raise RuntimeError(f"Accepted invalid archive: {change}")
        for selection, overrides in cases:
            settings = dict(base_url=base_url, token="unused")
            settings.update(overrides)
            try:
                upload_archives(selection, **settings)
            except ValueError:
                pass
            else:
                raise RuntimeError("Accepted invalid upload settings")
    print("PASS nine preflight rejections, before any network request")


def check_remote_content(rows, saved, selected_ids):
    """Match all saved records, allowing Logfire's column mapping and microsecond precision."""
    def attributes(items):
        """Decode the scalar OTLP attributes in these saved fixtures."""
        values = {}
        for item in items:
            kind, value = next(iter(item["value"].items()))
            values[item["key"]] = int(value) if kind == "intValue" else value
        return values

    def timestamp(value):
        """Compare at the precision returned by Logfire's SQL API."""
        seconds, nanos = divmod(int(value), 1_000_000_000)
        return datetime.fromtimestamp(seconds, timezone.utc) + timedelta(microseconds=nanos // 1000)

    matched = set()
    for run_id in selected_ids:
        for kind, records in saved[run_id].items():
            for record in records:
                candidates = [row for row in rows if row["trace_id"] == record["traceId"]]
                if kind == "spans":
                    candidates = [row for row in candidates if row["span_id"] == record["spanId"]]
                else:
                    candidates = [row for row in candidates
                                  if row["parent_span_id"] == record["spanId"]
                                  and row["log_body"] == record["body"]["stringValue"]]
                if len(candidates) != 1:
                    raise RuntimeError("Missing or duplicated span/native log")
                row = candidates[0]
                matched.add((row["trace_id"], row["span_id"]))
                if row["otel_resource_attributes"]["lits.run_id"] != run_id:
                    raise RuntimeError("Changed run identity")
                expected = attributes(record.get("attributes", []))
                for key, column in (("logfire.span_type", "kind"), ("logfire.level_num", "level"),
                                    ("logfire.msg", "message"), ("logfire.json_schema", "attributes_json_schema")):
                    if key in expected:
                        value = expected.pop(key)
                        if row[column] != (json.loads(value) if key == "logfire.json_schema" else value):
                            raise RuntimeError(f"Changed mapped attribute: {key}")
                if row["attributes"] != expected:
                    raise RuntimeError("Changed record attributes")
                start = record["startTimeUnixNano"] if kind == "spans" else record["timeUnixNano"]
                end = record["endTimeUnixNano"] if kind == "spans" else start
                for column, value in (("start_timestamp", start), ("end_timestamp", end)):
                    if datetime.fromisoformat(row[column].replace("Z", "+00:00")) != timestamp(value):
                        raise RuntimeError("Changed record time")
                if kind == "spans" and (row["span_name"] != record["name"]
                        or row["parent_span_id"] != record.get("parentSpanId")):
                    raise RuntimeError("Changed span name or parent")
    if len(matched) != len(rows):
        raise RuntimeError("Unexpected remote records")


def check_import(source_root, selected_root, credentials_dir, mode):
    """Compare saved originals/copies and optionally their hosted OTLP records.

    Native logs carry their enclosing span's ID, not a distinct log span ID.
    Logfire exposes that relationship as the log record's parent_span_id.
    Python logging forwarded through Logfire instead already exists as a span.
    """
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    source_root, selected_root = Path(source_root).resolve(), Path(selected_root).resolve()
    directories = sorted(path.parent for path in selected_root.rglob("archive.json"))
    if not directories:
        raise RuntimeError("No selected archives")
    originals = {p: p.read_bytes() for root in (source_root, selected_root)
                 for p in root.rglob("*") if p.is_file()}
    saved, selected_ids = {}, set()
    for marker in source_root.rglob("archive.json"):
        metadata = json.loads(marker.read_text())
        if metadata["status"] == "complete":
            saved[metadata["run_id"]] = read_archive(marker.parent, metadata["run_id"])
    for directory in directories:
        run_id = json.loads((directory / "archive.json").read_text())["run_id"]
        selected_ids.add(run_id)
        for filename in ("traces.jsonl", "logs.jsonl", "archive.json"):
            if (directory / filename).read_bytes() != (source_root / run_id / filename).read_bytes():
                raise RuntimeError("Relocated archive differs from original")
        for signal, message_type in (("traces", ExportTraceServiceRequest),
                                     ("logs", ExportLogsServiceRequest)):
            for line in (directory / f"{signal}.jsonl").read_text().splitlines():
                payload = json.loads(line)
                data, count = _encode_archive_batch(deepcopy(payload), signal, run_id)
                decoded = MessageToDict(message_type.FromString(data), use_integers_for_enums=True)
                resource_key, scope_key, record_key = (
                    ("resourceSpans", "scopeSpans", "spans") if signal == "traces"
                    else ("resourceLogs", "scopeLogs", "logRecords")
                )
                for resource in decoded[resource_key]:
                    for scope in resource[scope_key]:
                        for record in scope[record_key]:
                            for item in (record, *record.get("links", [])):
                                for key in ("traceId", "spanId", "parentSpanId"):
                                    if item.get(key):
                                        item[key] = base64.b64decode(item[key]).hex()
                if not count or decoded != payload:
                    raise RuntimeError("Protobuf round-trip changed OTLP content")
    check_preflight(directories)
    if mode == "offline":
        print("PASS saved-file selection and protobuf encoding; no upload requested")
        return

    from logfire.query_client import LogfireQueryClient

    credentials_dir = Path(credentials_dir)
    write = json.loads((credentials_dir / "logfire_credentials.json").read_text())
    read = json.loads((credentials_dir / "read_token.json").read_text())
    base_url = "https://logfire-us.pydantic.dev"
    if (write["project_url"] != base_url + "/xinzhel/lits" or read["base_url"] != base_url
            or read["project_name"] != "lits" or read["organization"] != "xinzhel"):
        raise RuntimeError("Unexpected credentials: this check targets xinzhel/lits")
    trace_ids = {span["traceId"] for records in saved.values() for span in records["spans"]}
    selected_traces = {span["traceId"] for run_id in selected_ids for span in saved[run_id]["spans"]}
    times = [int(span["startTimeUnixNano"]) / 1e9
             for records in saved.values() for span in records["spans"]]
    earliest = datetime.fromtimestamp(min(times), timezone.utc) - timedelta(seconds=1)
    latest = datetime.fromtimestamp(max(times), timezone.utc) + timedelta(seconds=1)
    sql = "SELECT * FROM records WHERE trace_id IN (" + ",".join(f"'{t}'" for t in sorted(trace_ids)) + ")"
    report_dir = Path(mkdtemp(prefix="lits-archive-import-"))
    report = {"selected_root": str(selected_root), "source_root": str(source_root), "mode": mode}
    print(f"Read-back report: {report_dir}", flush=True)
    with LogfireQueryClient(read_token=read["token"], base_url=base_url) as client:
        before = client.query_json_rows(sql, min_timestamp=earliest, max_timestamp=latest, limit=1000)["rows"]
        report["before"] = before
        (report_dir / "readback.json").write_text(json.dumps(report, indent=2))
        counts = {run_id: {"traces": len(saved[run_id]["spans"]),
                           "logs": len(saved[run_id]["logRecords"])} for run_id in selected_ids}
        if mode == "upload":
            if any(row["trace_id"] in selected_traces for row in before):
                raise RuntimeError("Selected traces already exist; refusing duplicate upload")
            counts = upload_archives(directories, base_url=base_url, token=write["token"])
            report["accepted"] = counts
            (report_dir / "readback.json").write_text(json.dumps(report, indent=2))
            # Allow ingestion to settle without exhausting the query API's minute quota.
            time.sleep(60)
            after = client.query_json_rows(sql, min_timestamp=earliest, max_timestamp=latest, limit=1000)["rows"]
        else:
            after = before
    report["after"] = after
    (report_dir / "readback.json").write_text(json.dumps(report, indent=2))
    selected = [row for row in after if row["trace_id"] in selected_traces]
    expected_count = sum(sum(count.values()) for count in counts.values())
    unselected = lambda rows: sorted(json.dumps(row, sort_keys=True) for row in rows
                                     if row["trace_id"] not in selected_traces)
    if len(selected) != expected_count or unselected(before) != unselected(after):
        raise RuntimeError("Selected count mismatch or unselected records changed")
    check_remote_content(selected, saved, selected_ids)
    for path, contents in originals.items():
        if path.read_bytes() != contents:
            raise RuntimeError("Upload changed an archive")
    print(f"PASS {counts}; full content, native-log linkage, unchanged local files")
    print(f"Unselected records: before={len(unselected(before))}, after={len(unselected(after))}")
    breakpoint()  # inspect: selected, saved, counts, report_dir


def main():
    """Require saved files; --upload sends them, --verify-only only queries Logfire."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--selected-root", required=True)
    parser.add_argument("--credentials-dir", default=".logfire")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--upload", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    # Preflight must finish without network even when a live upload is requested.
    network_allowed = False
    def guard(event, arguments):
        """Fail before any socket connection made without the explicit upload stage."""
        if not network_allowed and event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Unexpected network request")
    sys.addaudithook(guard)
    check_import(args.source_root, args.selected_root, args.credentials_dir, "offline")
    if args.upload or args.verify_only:
        network_allowed = True
        check_import(args.source_root, args.selected_root, args.credentials_dir,
                     "upload" if args.upload else "verify")


if __name__ == "__main__":
    main()
