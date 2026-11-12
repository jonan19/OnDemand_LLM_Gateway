"""Command-line streaming client for the gateway (task 2.7).

    python -m tools.chat_client "Say hi in 5 words"
    python -m tools.chat_client "And again?" --conversation-id 3 --url https://xyz.trycloudflare.com

Prints tokens as they arrive, then time-to-first-token and total time. Exit codes:
0 ok, 2 gateway says the worker is starting (HTTP 202, see docs/api-contract.md), 1 any other error.
Run from the gateway/ folder.
"""
import argparse
import json
import sys
import time

import httpx


def stream_chat(client: httpx.Client, url: str, message: str, conversation_id: int | None = None,
                max_tokens: int | None = None, out=sys.stdout, err=sys.stderr, clock=time.monotonic) -> int:
    body: dict = {"message": message}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    start = clock()
    first_token_at = None
    chunks = 0
    with client.stream("POST", url.rstrip("/") + "/chat", json=body) as resp:
        if resp.status_code == 202:
            info = json.loads(resp.read() or b"{}")
            print(f"Worker is starting (job {info.get('job_id')}, ~{info.get('estimated_wait_seconds')}s). "
                  "Re-send once it is ready.", file=err)
            return 2
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}: {resp.read().decode(errors='replace')}", file=err)
            return 1

        print(f"[conversation {resp.headers.get('X-Conversation-Id')}]", file=err)
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            if raw == "[DONE]":
                break
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "error" in payload:
                print(f"\n[stream error] {payload['error']}", file=err)
                return 1
            delta = (payload.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if first_token_at is None:
                    first_token_at = clock()
                chunks += 1
                out.write(delta)
                out.flush()

    total = clock() - start
    if chunks == 0:
        print(f"\n[no tokens received after {total:.2f}s: is the worker running and reachable from the gateway?]", file=err)
        return 1
    ttft = f"{first_token_at - start:.2f}s" if first_token_at is not None else "n/a"
    print(f"\n[first token: {ttft} | total: {total:.2f}s | {chunks} chunks]", file=err)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stream a chat reply from the gateway")
    p.add_argument("message")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="gateway base URL")
    p.add_argument("--conversation-id", type=int)
    p.add_argument("--max-tokens", type=int)
    args = p.parse_args(argv)
    try:
        with httpx.Client(timeout=httpx.Timeout(None, connect=10)) as client:
            return stream_chat(client, args.url, args.message, args.conversation_id, args.max_tokens)
    except httpx.ConnectError:
        print(f"Could not connect to {args.url}. Is the gateway running?", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
