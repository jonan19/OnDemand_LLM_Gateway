"""Task 3.7: the context sent to the model equals the last N stored turns."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main


def fill(db, conversation_id, count):
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        db.add_message(conversation_id, role, f"msg-{i}")


# --- storage level -----------------------------------------------------------

def test_returns_last_n_oldest_first(db):
    c = db.create_conversation()
    fill(db, c, 25)
    window = db.get_recent_messages(c, limit=10)
    assert [m["content"] for m in window] == [f"msg-{i}" for i in range(15, 25)]


def test_fewer_messages_than_limit_returns_all(db):
    c = db.create_conversation()
    fill(db, c, 3)
    assert [m["content"] for m in db.get_recent_messages(c, limit=10)] == ["msg-0", "msg-1", "msg-2"]


def test_empty_conversation_returns_empty(db):
    c = db.create_conversation()
    assert db.get_recent_messages(c, limit=10) == []


def test_exactly_n_messages(db):
    c = db.create_conversation()
    fill(db, c, 10)
    assert len(db.get_recent_messages(c, limit=10)) == 10


def test_window_is_per_conversation(db):
    a, b = db.create_conversation(), db.create_conversation()
    db.add_message(a, "user", "from-a")
    db.add_message(b, "user", "from-b")
    assert db.get_recent_messages(a, limit=10) == [{"role": "user", "content": "from-a"}]


def test_window_does_not_delete_older_history(db):
    c = db.create_conversation()
    fill(db, c, 25)
    db.get_recent_messages(c, limit=10)
    assert len(db.get_full_history(c)) == 25


def test_roles_are_preserved(db):
    c = db.create_conversation()
    fill(db, c, 4)
    assert [m["role"] for m in db.get_recent_messages(c, limit=4)] == ["user", "assistant", "user", "assistant"]


# --- /chat integration with a fake worker -----------------------------------

@pytest.fixture()
def chat_client(db, monkeypatch):
    """TestClient whose outbound worker calls are answered by a fake and recorded."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = (
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient

    class FakeWorkerClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main.httpx, "AsyncClient", FakeWorkerClient)
    monkeypatch.setattr(main, "SLIDING_WINDOW_TURNS", 4)
    return TestClient(main.app), sent


def test_chat_injects_only_last_n_stored_turns(db, chat_client):
    client, sent = chat_client
    c = db.create_conversation()
    fill(db, c, 10)  # msg-0 .. msg-9

    r = client.post("/chat", json={"message": "latest", "conversation_id": c})
    assert r.status_code == 200

    # Window of 4 = last 3 stored turns + the new user message just saved.
    assert [m["content"] for m in sent[0]["messages"]] == ["msg-7", "msg-8", "msg-9", "latest"]
    assert sent[0]["stream"] is True


def test_chat_stores_assistant_reply_after_stream(db, chat_client):
    client, _ = chat_client
    c = db.create_conversation()
    client.post("/chat", json={"message": "hi", "conversation_id": c})

    history = db.get_full_history(c)
    assert [(m["role"], m["content"]) for m in history] == [("user", "hi"), ("assistant", "Hello")]


def test_chat_returns_conversation_id_header_for_new_conversation(db, chat_client):
    client, _ = chat_client
    r = client.post("/chat", json={"message": "hi"})
    assert int(r.headers["X-Conversation-Id"]) >= 1


def test_older_turns_stay_in_storage_but_not_in_prompt(db, chat_client):
    client, sent = chat_client
    c = db.create_conversation()
    fill(db, c, 10)
    client.post("/chat", json={"message": "latest", "conversation_id": c})

    prompt_text = json.dumps(sent[0]["messages"])
    assert "msg-0" not in prompt_text
    assert any(m["content"] == "msg-0" for m in db.get_full_history(c))
