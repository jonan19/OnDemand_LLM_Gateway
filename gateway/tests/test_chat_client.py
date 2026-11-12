"""Task 2.7: the streaming test client, against a fake gateway."""
import io
import json

import httpx

from tools.chat_client import stream_chat

SSE = (
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    b"data: [DONE]\n\n"
)


def run(handler, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        code = stream_chat(client, "http://gw", "hi", out=out, err=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def test_streams_tokens_and_reports_timing():
    code, out, err = run(lambda r: httpx.Response(200, content=SSE, headers={"X-Conversation-Id": "5"}))
    assert code == 0
    assert out == "Hello"
    assert "[conversation 5]" in err and "first token" in err and "2 chunks" in err


def test_sends_message_and_optional_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["path"] = request.url.path
        return httpx.Response(200, content=SSE)

    run(handler, conversation_id=3, max_tokens=50)
    assert seen == {"message": "hi", "conversation_id": 3, "max_tokens": 50, "path": "/chat"}


def test_202_means_worker_starting_exit_code_2():
    body = {"job_id": "j_1", "state": "starting", "estimated_wait_seconds": 60}
    code, out, err = run(lambda r: httpx.Response(202, json=body))
    assert code == 2 and out == "" and "j_1" in err


def test_error_status_exit_code_1():
    code, _, err = run(lambda r: httpx.Response(503, json={"error": {"code": "worker_unavailable"}}))
    assert code == 1 and "503" in err


def test_mid_stream_error_event_exit_code_1():
    body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"error":{"code":"worker_unavailable"}}\n\n'
    code, out, err = run(lambda r: httpx.Response(200, content=body))
    assert code == 1 and out == "Hi" and "stream error" in err


def test_empty_stream_is_an_error():
    code, out, err = run(lambda r: httpx.Response(200, content=b"data: [DONE]\n\n"))
    assert code == 1 and out == "" and "no tokens received" in err
