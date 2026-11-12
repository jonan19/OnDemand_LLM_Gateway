"""Task 4.9 / 4.12: worker state machine transitions."""
import itertools

import pytest

from state import ALLOWED_TRANSITIONS, InvalidTransition, WorkerState, WorkerStateMachine

S = WorkerState


def test_starts_idle_by_default():
    assert WorkerStateMachine().state is S.IDLE


def test_happy_path_idle_starting_ready_idle():
    m = WorkerStateMachine()
    assert m.start() is S.IDLE
    assert m.state is S.STARTING
    assert m.mark_ready() is S.STARTING
    assert m.state is S.READY
    assert m.mark_idle() is S.READY
    assert m.state is S.IDLE


def test_starting_can_fall_back_to_idle_when_launch_fails():
    m = WorkerStateMachine()
    m.start()
    m.mark_idle()
    assert m.state is S.IDLE
    m.start()  # a retry launches again
    assert m.state is S.STARTING


@pytest.mark.parametrize("current,new", [
    (a, b) for a, b in itertools.product(S, S) if b not in ALLOWED_TRANSITIONS[a]
])
def test_every_disallowed_transition_raises_and_keeps_state(current, new):
    m = WorkerStateMachine(initial=current)
    with pytest.raises(InvalidTransition) as err:
        m.transition(new)
    assert err.value.current is current and err.value.requested is new
    assert m.state is current


@pytest.mark.parametrize("current,new", [
    (a, b) for a in S for b in ALLOWED_TRANSITIONS[a]
])
def test_every_allowed_transition_works(current, new):
    m = WorkerStateMachine(initial=current)
    assert m.can_transition(new)
    m.transition(new)
    assert m.state is new


def test_ready_cannot_jump_to_starting():
    """A crashed worker must be torn down (IDLE) before relaunching."""
    m = WorkerStateMachine(initial=S.READY)
    assert not m.can_transition(S.STARTING)


def test_seconds_in_state_resets_on_transition():
    now = [100.0]
    m = WorkerStateMachine(clock=lambda: now[0])
    now[0] = 130.0
    assert m.seconds_in_state == 30.0
    m.start()
    assert m.seconds_in_state == 0.0
    now[0] = 145.0
    assert m.seconds_in_state == 15.0


def test_failed_transition_does_not_reset_timer():
    now = [0.0]
    m = WorkerStateMachine(clock=lambda: now[0])
    now[0] = 10.0
    with pytest.raises(InvalidTransition):
        m.mark_ready()
    assert m.seconds_in_state == 10.0


def test_values_are_lowercase_strings_for_the_api():
    assert [s.value for s in S] == ["idle", "starting", "ready"]
    assert S.READY == "ready"
