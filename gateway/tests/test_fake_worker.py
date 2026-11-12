"""The fake worker behaves like llama-server for the parts the gateway uses."""
import json

from fastapi.testclient import TestClient

from tools.fake_worker import create_app


def chunks(resp):
    out = []
    for line in resp.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


def test_health_ok_when_not_loading():
    r = TestClient(create_app()).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_health_503_while_loading_then_200():
    now = [0.0]
    client = TestClient(create_app(load_delay=10, clock=lambda: now[0]))
    assert client.get("/health").status_code == 503
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 503
    now[0] = 10.0
    assert client.get("/health").status_code == 200


def test_streams_an_echo_one_word_at_a_time():
    client = TestClient(create_app(token_delay=0))
    body = {"messages": [{"role": "user", "content": "hello big world"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks(resp))
    assert text == "Echo: hello big world"


def test_stream_ends_with_done():
    client = TestClient(create_app(token_delay=0))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert r.text.rstrip().endswith("data: [DONE]")


def test_non_streaming_reply_includes_timings():
    client = TestClient(create_app(token_delay=0.01))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert r["choices"][0]["message"]["content"] == "Echo: hi"
    assert r["timings"]["predicted_per_second"] > 0


def test_max_tokens_limits_the_reply():
    client = TestClient(create_app(token_delay=0))
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "a b c d e"}], "max_tokens": 2}).json()
    assert r["choices"][0]["message"]["content"] == "Echo: a"


def test_debug_last_shows_the_context_it_received():
    client = TestClient(create_app(token_delay=0))
    msgs = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"}]
    client.post("/v1/chat/completions", json={"messages": msgs})
    assert client.get("/debug/last").json()["messages"] == msgs
