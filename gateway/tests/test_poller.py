"""Task 4.11 / 4.12: health poller against a mocked worker, with a fake clock (no real waiting)."""
import asyncio

import httpx
import pytest

from poller import WorkerStartTimeout, wait_until_healthy


class FakeTime:
    """Fake clock + sleep: sleeping just advances the clock and records the delay."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responses, seen=None):
    """`responses`: list of status codes or Exceptions, consumed one per request; the last repeats."""
    queue = list(responses)

    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(str(request.url))
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(responses, seen=None, **kwargs):
    t = FakeTime()

    async def go():
        async with make_client(responses, seen) as client:
            return await wait_until_healthy(
                "http://worker:8080", client=client, sleep=t.sleep, clock=t.clock, **kwargs
            )

    return t, go


def test_ready_immediately_returns_without_sleeping():
    t, go = run([200])
    assert asyncio.run(go()) == 0
    assert t.sleeps == []


def test_backoff_sequence_is_1_2_4_then_capped_at_4():
    t, go = run([503, 503, 503, 503, 503, 200])
    waited = asyncio.run(go())
    assert t.sleeps == [1, 2, 4, 4, 4]
    assert waited == 15


def test_503_while_loading_is_not_ready():
    t, go = run([503, 200])
    asyncio.run(go())
    assert t.sleeps == [1]


def test_connection_errors_count_as_not_ready():
    t, go = run([httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 200])
    asyncio.run(go())
    assert t.sleeps == [1, 2]


def test_only_200_counts_as_healthy():
    t, go = run([404, 500, 204, 200])
    asyncio.run(go())
    assert len(t.sleeps) == 3


def test_times_out_after_the_limit_and_reports_attempts():
    t, go = run([503], timeout=300)
    with pytest.raises(WorkerStartTimeout) as err:
        asyncio.run(go())
    assert t.now == 300  # never sleeps past the deadline
    assert err.value.waited == 300
    # 1 + 2 + 4 + (4 * 73) = 299, then a final clamped sleep of 1 reaches exactly 300
    assert err.value.attempts == len(t.sleeps) + 1


def test_last_sleep_is_clamped_to_the_remaining_time():
    t, go = run([503], timeout=10)
    with pytest.raises(WorkerStartTimeout):
        asyncio.run(go())
    assert t.sleeps == [1, 2, 4, 3]
    assert sum(t.sleeps) == 10


def test_one_last_check_happens_right_at_the_deadline():
    # Becomes healthy exactly on the final attempt at t=10; must succeed, not time out.
    t, go = run([503, 503, 503, 503, 200], timeout=10)
    assert asyncio.run(go()) == 10


def test_polls_the_health_path_and_tolerates_trailing_slash():
    seen = []
    t = FakeTime()

    async def go():
        async with make_client([200], seen) as client:
            await wait_until_healthy("http://worker:8080/", client=client, sleep=t.sleep, clock=t.clock)

    asyncio.run(go())
    assert seen == ["http://worker:8080/health"]


def test_custom_intervals():
    t, go = run([503, 503, 503, 200], first_interval=0.5, max_interval=1)
    asyncio.run(go())
    assert t.sleeps == [0.5, 1, 1]
