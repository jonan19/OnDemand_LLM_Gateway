"""Worker lifecycle state machine (task 4.9).

    IDLE --launch--> STARTING --healthy--> READY --idle timeout--> IDLE
                        |
                        +--failed / gave up--> IDLE

Anything else is invalid. In particular READY never goes straight back to STARTING: a crashed
worker is first torn down (READY -> IDLE) and then relaunched (IDLE -> STARTING), so every
launch goes through the same path.

Values are lowercase strings so they can be returned as-is in the API (`worker_state` on
/health, `state` on /status/{job_id}; see docs/api-contract.md).

This module has no I/O. Persisting the state (task 4.10) and acting on transitions
(provisioner, queue, idle timer) are layered on top.
"""
import time
from enum import Enum
from typing import Callable


class WorkerState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"


ALLOWED_TRANSITIONS: dict[WorkerState, frozenset[WorkerState]] = {
    WorkerState.IDLE: frozenset({WorkerState.STARTING}),
    WorkerState.STARTING: frozenset({WorkerState.READY, WorkerState.IDLE}),
    WorkerState.READY: frozenset({WorkerState.IDLE}),
}


class InvalidTransition(Exception):
    def __init__(self, current: WorkerState, requested: WorkerState):
        self.current = current
        self.requested = requested
        super().__init__(f"Invalid worker state transition: {current.value} -> {requested.value}")


class WorkerStateMachine:
    def __init__(
        self,
        initial: WorkerState = WorkerState.IDLE,
        clock=time.monotonic,
        on_change: Callable[[WorkerState, WorkerState], None] | None = None,
    ):
        """`on_change(previous, new)` runs after every successful transition (used for persistence)."""
        self._clock = clock
        self._on_change = on_change
        self._state = initial
        self._since = clock()

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def seconds_in_state(self) -> float:
        """How long the machine has been in the current state (e.g. for elapsed_seconds)."""
        return self._clock() - self._since

    def can_transition(self, new: WorkerState) -> bool:
        return new in ALLOWED_TRANSITIONS[self._state]

    def transition(self, new: WorkerState) -> WorkerState:
        """Move to `new` or raise InvalidTransition. Returns the previous state."""
        if not self.can_transition(new):
            raise InvalidTransition(self._state, new)
        previous = self._state
        self._state = new
        self._since = self._clock()
        if self._on_change is not None:
            self._on_change(previous, new)
        return previous

    # Readable shortcuts for the events the lifecycle code will actually fire.
    def start(self) -> WorkerState:
        return self.transition(WorkerState.STARTING)

    def mark_ready(self) -> WorkerState:
        return self.transition(WorkerState.READY)

    def mark_idle(self) -> WorkerState:
        return self.transition(WorkerState.IDLE)


def persistent_machine(
    load: Callable[[], str | None],
    save: Callable[[str], None],
    clock=time.monotonic,
) -> WorkerStateMachine:
    """A state machine that survives gateway restarts (task 4.10).

    `load()` returns the last saved state string or None; `save(state)` stores it. In the gateway
    these are storage.get_worker_state / storage.set_worker_state. An unknown or missing saved
    value starts at IDLE. Every successful transition is saved.

    A restored STARTING or READY is only what the gateway *believed* before it stopped; the worker
    may be gone. Callers should check the worker's /health on startup and call mark_idle() if it
    does not answer (done in Phase 5 when the lifecycle is wired up).
    """
    saved = load()
    try:
        initial = WorkerState(saved) if saved is not None else WorkerState.IDLE
    except ValueError:
        initial = WorkerState.IDLE
    return WorkerStateMachine(initial=initial, clock=clock, on_change=lambda _prev, new: save(new.value))
