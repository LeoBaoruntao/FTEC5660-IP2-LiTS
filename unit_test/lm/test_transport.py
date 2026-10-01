"""Exercise transport.py::request_with_network_retry using real loopback connections.

Run PYTHONBREAKPOINT=0 python -m unit_test.lm.test_transport. No model inference:
a delayed local HTTP server verifies recovery through HTTPX and the OpenAI SDK.
Check CRITICAL messages in both console and file, HTTP rejection, and cancellation.
"""

from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from threading import Thread, Timer

import httpx
from openai import AuthenticationError, OpenAI

from lits.lm.transport import request_with_network_retry
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
    print("Passed HTTPX/SDK recovery, authentication failure, CRITICAL logs, and cancellation checks")


if __name__ == "__main__":
    main()
