"""Worker health poller (task 4.11).

After a worker is launched it needs 30-90s to load the model. Until then its /health endpoint
either refuses connections or answers 503 (llama-server returns 503 while loading and 200 once
ready). This module polls that endpoint with backoff until it returns 200, or gives up.

Defaults follow the spec: 1s, 2s, 4s, 4s, ... between polls, and a 5 minute overall limit.
No state-machine or provisioner logic lives here; the caller decides what a success or a
timeout means (READY / reprovision).
"""
import asyncio
import time
from typing import Awaitable, Callable

import httpx

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_FIRST_INTERVAL = 1.0
DEFAULT_MAX_INTERVAL = 4.0
REQUEST_TIMEOUT_SECONDS = 5.0


class WorkerStartTimeout(Exception):
    """The worker did not report healthy within the allowed time."""

    def __init__(self, waited: float, attempts: int):
        self.waited = waited
        self.attempts = attempts
        super().__init__(f"Worker not healthy after {waited:.0f}s ({attempts} attempts)")


async def wait_until_healthy(
    worker_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    first_interval: float = DEFAULT_FIRST_INTERVAL,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Poll `<worker_url>/health` until it returns HTTP 200.

    Returns the seconds waited. Raises WorkerStartTimeout if `timeout` seconds pass first.
    Connection errors, timeouts and non-200 responses all count as "not ready yet".
    `client`, `sleep` and `clock` are injectable for tests.
    """
    url = worker_url.rstrip("/") + "/health"
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    start = clock()
    deadline = start + timeout
    interval = first_interval
    attempts = 0
    try:
        while True:
            attempts += 1
            try:
                # Never let a hanging request outlast the overall deadline.
                per_request = max(min(REQUEST_TIMEOUT_SECONDS, deadline - clock()), 0.1)
                resp = await client.get(url, timeout=per_request)
                if resp.status_code == 200:
                    return clock() - start
            except httpx.HTTPError:
                pass  # refused / timed out / reset: the worker simply isn't up yet

            remaining = deadline - clock()
            if remaining <= 0:
                raise WorkerStartTimeout(clock() - start, attempts)
            await sleep(min(interval, remaining))
            interval = min(interval * 2, max_interval)
    finally:
        if owns_client:
            await client.aclose()
