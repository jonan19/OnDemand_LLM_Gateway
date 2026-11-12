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
    def __init__(self, initial: WorkerState = WorkerState.IDLE, clock=time.monotonic):
        self._clock = clock
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
        return previous

    # Readable shortcuts for the events the lifecycle code will actually fire.
    def start(self) -> WorkerState:
        return self.transition(WorkerState.STARTING)

    def mark_ready(self) -> WorkerState:
        return self.transition(WorkerState.READY)

    def mark_idle(self) -> WorkerState:
        return self.transition(WorkerState.IDLE)
