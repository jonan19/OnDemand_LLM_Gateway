"""A fake llama-server for local testing, no model or GPU needed.

    python -m tools.fake_worker                      # listens on 127.0.0.1:8081
    python -m tools.fake_worker --load-delay 20      # /health answers 503 for the first 20s (cold start)
    python -m tools.fake_worker --token-delay 0.3    # slower streaming
    python -m tools.fake_worker --fail-after 3       # drop the connection after 3 tokens (mid-stream crash)

Then point the gateway at it: WORKER_URL=http://127.0.0.1:8081 in gateway/.env (port 8081 because
8080 is taken by Apache on the dev machine). Run from the gateway/ folder.

It mimics the parts of llama-server the gateway uses:
  GET  /health                 200 {"status":"ok"}, or 503 while "loading"
  POST /v1/chat/completions    OpenAI-style; streams SSE chunks when stream=true
Extras for debugging:
  GET  /debug/last             the last request body received (see exactly what context the gateway sent)

The reply is deterministic: "Echo: <the last user message>", streamed one word at a time.
"""
import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse


def create_app(load_delay: float = 0.0, token_delay: float = 0.05, fail_after: int | None = None,
               clock=time.monotonic) -> FastAPI:
    app = FastAPI(title="Fake llama-server")
    started = clock()
    state = {"last_request": None}

    def loading() -> bool:
        return clock() - started < load_delay

    def loading_response():
        return JSONResponse(status_code=503, content={"error": {"code": 503, "message": "Loading model"}})

    @app.get("/health")
    async def health():
        return loading_response() if loading() else {"status": "ok"}

    @app.get("/debug/last")
    async def last_request():
        return state["last_request"] or {}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if loading():
            return loading_response()
        state["last_request"] = body
        messages = body.get("messages", [])
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        words = f"Echo: {last_user}".split()
        max_tokens = body.get("max_tokens")
        if max_tokens:
            words = words[:max_tokens]

        if not body.get("stream"):
            return {
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": " ".join(words)}}],
                "timings": {"prompt_per_second": 1000.0, "predicted_per_second": 1 / max(token_delay, 0.001)},
            }

        async def stream():
            for i, word in enumerate(words):
                if fail_after is not None and i >= fail_after:
                    raise RuntimeError("fake worker crashed mid-stream (--fail-after)")
                await asyncio.sleep(token_delay)
                text = word if i == 0 else " " + word
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': text}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Fake llama-server for local gateway testing")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--load-delay", type=float, default=0.0, help="seconds /health returns 503 after start")
    p.add_argument("--token-delay", type=float, default=0.05, help="seconds between streamed words")
    p.add_argument("--fail-after", type=int, help="crash the stream after this many tokens")
    args = p.parse_args(argv)
    print(f"Fake worker on http://{args.host}:{args.port} "
          f"(load-delay={args.load_delay}s, token-delay={args.token_delay}s, fail-after={args.fail_after})")
    uvicorn.run(create_app(args.load_delay, args.token_delay, args.fail_after),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
