"""Test setup: point storage at a throwaway DB *before* main is imported.

main.py calls storage.init_db() at import time, so DB_PATH must be redirected first or the
real gateway.db would be touched. main also serves ./static relative to the CWD, so we chdir
to the gateway folder.
"""
import os
import pathlib
import tempfile

GATEWAY_DIR = pathlib.Path(__file__).resolve().parent.parent
os.chdir(GATEWAY_DIR)

import storage  # noqa: E402

_TMP_DIR = tempfile.mkdtemp(prefix="gateway-tests-")
storage.DB_PATH = pathlib.Path(_TMP_DIR) / "import.db"

import pytest  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Fresh empty database for each test."""
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()
    return storage
