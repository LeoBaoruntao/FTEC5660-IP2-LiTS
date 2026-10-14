"""Exercise transport.py::request_with_network_retry using real loopback connections.

Run PYTHONBREAKPOINT=0 python -m unit_test.lm.test_transport. No model inference:
a delayed local HTTP server verifies recovery through HTTPX and the OpenAI SDK.
Check CRITICAL messages in both console and file, HTTP rejection, and cancellation.
"""

from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from threading import Thread, Timer

import httpx
from openai import APIStatusError, AuthenticationError, OpenAI

from lits.lm.transport import request_with_network_retry
from lits.lm.openai_chat import OpenAIChatModel
from lits.lm.openai_completion import OpenAICompletionModel
from lits.log import setup_logging


class Handler(BaseHTTPRequestHandler):
    """Serve an empty API collection or an explicit authentication rejection."""

    def do_GET(self):
        """Return valid JSON without any model or external service."""
        payload = b'{"object":"list","data":[]}'
        self.send_response(401 if self.path.endswith("denied") else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        """Keep HTTP access chatter out of the recovery log checks."""
        return


class GatewayHandler(Handler):
    """Serve configured HTTP failures followed by a successful metadata response."""

    def do_GET(self):
        """Count real requests and consume the server's configured status sequence."""
        self.server.calls += 1
        status = self.server.statuses.pop(0) if self.server.statuses else 200
        payload = b'{"object":"list","data":[]}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        """Record actual SDK payloads and reject generation without running a model."""
        self.server.calls += 1
        self.server.payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = b'{"error":{"message":"Temporary local test failure","type":"server_error"}}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def check_configurable_deadline():
    """Check environment/model precedence, unlimited recovery and cancellation over local HTTP."""
    variable = "LITS_NETWORK_RECOVERY_TIMEOUT"
    original = os.environ.get(variable)
    server = HTTPServer(("127.0.0.1", 0), GatewayHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with OpenAI(api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}",
                    http_client=httpx.Client(trust_env=False, timeout=2), max_retries=0) as api:
            for invalid in ("-1", "nan", "-inf", "invalid"):
                os.environ[variable] = invalid
                server.calls = 0
                try:
                    request_with_network_retry(api.models.list)
                except ValueError:
                    if server.calls:
                        raise RuntimeError("Invalid recovery timeout reached the server")
                else:
                    raise RuntimeError("Invalid recovery timeout was accepted")
            os.environ[variable] = "inf"
            server.statuses, server.calls = [503, 200], 0
            response = request_with_network_retry(api.models.list)
            if response.data != [] or server.calls != 2:
                raise RuntimeError("Unlimited metadata recovery failed")

        for model_type in (OpenAIChatModel, OpenAICompletionModel):
            # Explicit finite budgets override unlimited defaults; None reads the environment.
            for environment, timeout in (("inf", 0.05), ("0.05", None)):
                os.environ[variable] = environment
                model = model_type("local-test", api_key="local-test",
                                   base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                   recovery_timeout=timeout)
                model.client.max_retries = 0
                server.calls = 0
                try:
                    model("Local transport test", max_new_tokens=1)
                except SystemExit:
                    if server.calls != 1 or "recovery_timeout" in server.payload:
                        raise RuntimeError("Model timeout was ignored or sent to the API")
                else:
                    raise RuntimeError("Model recovery did not exhaust its finite budget")
                finally:
                    model.client.close()

            # An explicit unlimited model budget overrides a zero environment budget.
            os.environ[variable] = "0"
            model = model_type("local-test", api_key="local-test",
                               base_url=f"http://127.0.0.1:{server.server_port}/v1",
                               recovery_timeout=float("inf"))
            model.client.max_retries = 0
            timer = Timer(0.5, os.kill, args=(os.getpid(), signal.SIGINT))
            timer.start()
            try:
                model("Local transport test", max_new_tokens=1)
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("Unlimited recovery did not remain cancellable")
            finally:
                timer.cancel()
                timer.join()
                model.client.close()
        breakpoint()  # inspect: server.calls, server.payload (no recovery option sent to API)
    finally:
        if original is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = original
        server.shutdown()
        worker.join()
        server.server_close()


def check_gateway_recovery():
    """Check SDK server/gateway retries, status filtering and bounded recovery without inference."""
    server = HTTPServer(("127.0.0.1", 0), GatewayHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with OpenAI(api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}",
                    http_client=httpx.Client(trust_env=False, timeout=2), max_retries=0) as api:
            for status in (500, 502, 503, 504):
                server.statuses, server.calls = [status, 200], 0
                response = request_with_network_retry(api.models.list)
                if response.data != [] or server.calls != 2:
                    raise RuntimeError(f"HTTP {status} recovery failed")
            for status in (400, 401, 429, 501):
                server.statuses, server.calls = [status], 0
                try:
                    request_with_network_retry(api.models.list)
                except APIStatusError as error:
                    if error.status_code != status or server.calls != 1:
                        raise RuntimeError(f"HTTP {status} handling changed") from error
                else:
                    raise RuntimeError(f"HTTP {status} did not propagate")
            server.statuses, server.calls = [500], 0
            try:
                request_with_network_retry(api.models.list, recovery_timeout=0.05)
            except SystemExit:
                if server.calls != 1:
                    raise RuntimeError("A gateway retry started after the recovery deadline")
            else:
                raise RuntimeError("Gateway recovery exhaustion did not stop execution")
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def check_recovery(use_sdk):
    """Refuse the first TCP connection, then accept the same retried request."""
    server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
    server.server_bind()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    worker = Thread(target=server.serve_forever, daemon=True)

    def start_server():
        """Make the reserved socket reachable after the initial request fails."""
        server.server_activate()
        worker.start()

    timer = Timer(6 if use_sdk else 1, start_server)
    try:
        with httpx.Client(trust_env=False, timeout=0.1) as client:
            if use_sdk:
                api = OpenAI(api_key="local-test", base_url=url, http_client=client, max_retries=0)
                request = api.models.list
                timer.start()
                response = request_with_network_retry(request)
                if response.data != []:
                    raise RuntimeError("SDK response changed")
                try:
                    request_with_network_retry(api.get, "/denied", cast_to=dict)
                except AuthenticationError:
                    pass
                else:
                    raise RuntimeError("Authentication failure did not propagate")
            else:
                timer.start()
                response = request_with_network_retry(client.get, url)
                if response.status_code != 200:
                    raise RuntimeError("HTTPX recovery failed")
    finally:
        timer.join()
        server.shutdown()
        worker.join()
        server.server_close()


def main():
    """Check real recovery, unchanged HTTP errors, logging and interruption during wait."""
    console = StringIO()
    with TemporaryDirectory(prefix="lits-network-retry-") as directory, redirect_stderr(console):
        setup_logging("execution", directory, add_console_handler=True)
        check_recovery(False)
        check_recovery(True)
        # A bound but non-listening socket reliably refuses a request.
        server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
        server.server_bind()
        client = httpx.Client(trust_env=False, timeout=0.1)
        timer = Timer(1, os.kill, args=(os.getpid(), signal.SIGINT))
        timer.start()
        try:
            request_with_network_retry(client.get, f"http://127.0.0.1:{server.server_address[1]}")
        except KeyboardInterrupt:
            pass
        else:
            raise RuntimeError("Cancellation did not propagate")
        finally:
            timer.cancel()
            timer.join()
            client.close()
            server.server_close()
        with httpx.Client(trust_env=False, timeout=2) as client:
            try:
                request_with_network_retry(client.get, f"http://127.0.0.1:{server.server_address[1]}",
                                           recovery_timeout=0.05)
            except SystemExit:
                pass
            else:
                raise RuntimeError("Recovery exhaustion did not terminate the experiment")
        for text in (console.getvalue(), (Path(directory) / "execution.log").read_text()):
            if (text.count("[CRITICAL]") != 8 or text.count("connection recovered") != 2
                    or "retry 2 in 10.0 seconds" not in text
                    or "local-test" in text):
                raise RuntimeError(f"Unexpected recovery log: {text.replace('local-test', '<redacted>')}")
        breakpoint()  # inspect: console.getvalue(), directory
        check_gateway_recovery()
        check_configurable_deadline()
        if "Recovery budget: unlimited" not in console.getvalue():
            raise RuntimeError("Unlimited recovery was not identified in the log")
    print("Passed HTTPX/SDK recovery, authentication failure, CRITICAL logs, and cancellation checks")
    print("Passed HTTP 500/502/503/504 recovery, other-status propagation, and server deadline checks")
    print("Passed configurable deadlines, unlimited recovery and chat/completion parameter checks")


if __name__ == "__main__":
    main()
