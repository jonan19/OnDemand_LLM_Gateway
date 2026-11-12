"""Task 4.10: worker state survives restarts."""
import sqlite3

import pytest

from state import InvalidTransition, WorkerState, persistent_machine

S = WorkerState


def machine(db):
    return persistent_machine(db.get_worker_state, db.set_worker_state)


# --- storage ---------------------------------------------------------------

def test_nothing_saved_yet_returns_none(db):
    assert db.get_worker_state() is None


def test_save_and_load_roundtrip(db):
    db.set_worker_state("starting")
    assert db.get_worker_state() == "starting"


def test_saving_overwrites_a_single_row(db):
    for s in ("starting", "ready", "idle"):
        db.set_worker_state(s)
    assert db.get_worker_state() == "idle"
    conn = sqlite3.connect(db.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM worker_state").fetchone()[0] == 1
    conn.close()


def test_init_db_adds_the_table_to_an_existing_database(db):
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("DROP TABLE worker_state")
    conn.commit()
    conn.close()
    db.init_db()  # what happens when an old gateway.db is opened by the new code
    db.set_worker_state("ready")
    assert db.get_worker_state() == "ready"


# --- persistent machine ----------------------------------------------------

def test_fresh_database_starts_idle_and_writes_nothing(db):
    assert machine(db).state is S.IDLE
    assert db.get_worker_state() is None


def test_every_transition_is_saved(db):
    m = machine(db)
    m.start()
    assert db.get_worker_state() == "starting"
    m.mark_ready()
    assert db.get_worker_state() == "ready"
    m.mark_idle()
    assert db.get_worker_state() == "idle"


def test_state_survives_a_restart(db):
    m = machine(db)
    m.start()
    m.mark_ready()
    restarted = machine(db)  # new process, same database file
    assert restarted.state is S.READY
    restarted.mark_idle()  # and it keeps working from the restored state
    assert machine(db).state is S.IDLE


def test_invalid_transition_is_not_saved(db):
    m = machine(db)
    m.start()
    with pytest.raises(InvalidTransition):
        m.transition(S.STARTING)
    assert db.get_worker_state() == "starting"


def test_garbage_in_the_database_falls_back_to_idle(db):
    db.set_worker_state("exploded")
    assert machine(db).state is S.IDLE


def test_plain_machine_without_persistence_still_works():
    from state import WorkerStateMachine
    m = WorkerStateMachine()
    m.start()
    assert m.state is S.STARTING
