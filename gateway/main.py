import json
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

import storage

# Config comes from environment variables (see .env.example). Run with:
#   uvicorn main:app --env-file .env
WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:8080").rstrip("/")
SLIDING_WINDOW_TURNS = int(os.getenv("SLIDING_WINDOW_TURNS", "10"))  # last N stored messages sent per prompt
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))

app = FastAPI(title="Personal LLM Gateway")
storage.init_db()


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    conversation_id: int | None = None


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{WORKER_URL}/health", timeout=5)
            try:
                worker = r.json()
            except ValueError:
                # Something answered, but it isn't a llama-server health response.
                worker = {"error": f"non-JSON response (HTTP {r.status_code})"}
            return {"gateway": "ok", "worker": worker}
        except httpx.RequestError:
            return {"gateway": "ok", "worker": "unreachable"}


@app.get("/conversations")
async def list_conversations():
    return {"conversations": storage.list_conversations()}


@app.post("/conversations")
async def new_conversation():
    conversation_id = storage.create_conversation()
    return {"conversation_id": conversation_id}


@app.get("/conversations/{conversation_id}")
async def conversation_history(conversation_id: int):
    if not storage.conversation_exists(conversation_id):
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "not_found", "message": "Conversation not found", "retryable": False}},
        )
    return {"messages": storage.get_full_history(conversation_id)}


@app.post("/chat")
async def chat(req: ChatRequest):
    conversation_id = req.conversation_id
    if conversation_id is None or not storage.conversation_exists(conversation_id):
        conversation_id = storage.create_conversation()

    storage.add_message(conversation_id, "user", req.message)

    # Sliding window: only the last N stored turns go to the model.
    # Older turns stay in SQLite but are not sent with the prompt.
    history = storage.get_recent_messages(conversation_id, limit=SLIDING_WINDOW_TURNS)

    payload = {
        "messages": history,
        "max_tokens": req.max_tokens,
        "stream": True,
    }

    async def stream_response():
        assistant_chunks: list[str] = []
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{WORKER_URL}/v1/chat/completions", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    yield (line + "\n\n").encode()
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        continue
                    try:
                        delta = json.loads(raw)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        delta = None
                    if delta:
                        assistant_chunks.append(delta)

        full_text = "".join(assistant_chunks)
        if full_text:
            storage.add_message(conversation_id, "assistant", full_text)

    response = StreamingResponse(stream_response(), media_type="text/event-stream")
    response.headers["X-Conversation-Id"] = str(conversation_id)
    return response


app.mount("/", StaticFiles(directory="static", html=True), name="static")
