"""Shared connection recovery for synchronous model API requests."""

import logging
import math
import os
import time

import httpx
from openai import APIConnectionError, APIStatusError


logger = logging.getLogger(__name__)


def request_with_network_retry(request, *args, recovery_timeout=None, **kwargs):
    """Recover connectivity within a configurable deadline, retrying identical arguments.

    recovery_timeout is nonnegative seconds; float("inf") retries indefinitely.
    None reads LITS_NETWORK_RECOVERY_TIMEOUT (default 3600; "inf" is unlimited).
    Explicit values override the environment. Zero allows only the initial attempt.

    Retry connection failures, timeouts, interrupted remote responses and HTTP
    500/502/503/504 server or gateway failure responses after
    5, 10, 20, 40, then 60 seconds (capped). Log once per retry. recovery_timeout
    is seconds measured from the initial request;
    no further attempt starts after this deadline. An in-flight request may run
    until its own timeout before the deadline is checked. Existing SDK retries
    run first. Other HTTP status errors and response
    parsing errors propagate; KeyboardInterrupt/SystemExit always propagate.
    CRITICAL retry/recovery messages reach LiTS's configured console and file
    handlers. Never log request arguments, credentials, or exception messages.
    On exhaustion, SystemExit terminates the experiment, bypassing component and
    example-level Exception handlers so no fallback score or next example occurs.

    A response lost after server execution may cause duplicate work or billing.
    The caller records only the successfully received response's token usage;
    its elapsed request time includes reconnect waits.
    """
    recovery_timeout = float(os.getenv("LITS_NETWORK_RECOVERY_TIMEOUT", "3600")
                             if recovery_timeout is None else recovery_timeout)
    if math.isnan(recovery_timeout) or recovery_timeout < 0:
        raise ValueError("recovery_timeout must be nonnegative seconds or inf")
    failures = 0
    retry_delay = 5
    deadline = time.monotonic() + recovery_timeout
    while True:
        if failures and time.monotonic() >= deadline:
            logger.critical("Network recovery exhausted after %.0f seconds; terminating experiment. "
                            "Saved results are preserved.", recovery_timeout)
            raise SystemExit("Network recovery deadline exceeded; experiment stopped.")
        try:
            response = request(*args, **kwargs)
        except (APIConnectionError, APIStatusError, httpx.NetworkError, httpx.TimeoutException,
                httpx.RemoteProtocolError) as error:
            if isinstance(error, APIStatusError) and error.status_code not in (500, 502, 503, 504):
                raise
            if isinstance(error.__cause__, (httpx.UnsupportedProtocol, httpx.LocalProtocolError,
                                           httpx.InvalidURL)):
                raise
            failures += 1
            remaining = max(0, deadline - time.monotonic())
            delay = min(retry_delay, remaining)
            budget = "unlimited" if math.isinf(remaining) else f"{remaining:.0f} seconds remaining"
            logger.critical(
                "Model connection interrupted (%s); retry %d in %.1f seconds. "
                "Recovery budget: %s; Ctrl-C cancels. "
                "No response or score substituted.",
                type(error).__name__, failures, delay, budget,
            )
            time.sleep(delay)
            retry_delay = min(retry_delay * 2, 60)
        else:
            if failures:
                logger.critical("Model connection recovered after %d retries; continuing.", failures)
            return response
