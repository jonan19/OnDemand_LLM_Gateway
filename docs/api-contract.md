# Gateway API Contract (task 0.6)

The agreed interface between the **gateway** (Person A) and the **UI / clients / worker lifecycle** (Person B).
Both sides code against this document. Change it here first, then in code.

Legend: **[exists]** = implemented in `gateway/main.py` today. **[new]** = to be built (Phase 3 or 5).
Decisions made are recorded in s5.

Base URL: the gateway (`http://127.0.0.1:8000` locally, the permanent URL once deployed). The worker is never
called directly by clients.

---

## 1. Conventions

- Bodies are JSON (`Content-Type: application/json`) except streaming responses.
- Timestamps are ISO 8601 UTC strings (e.g. `2026-11-12T03:40:41+00:00`).
- Conversation IDs are integers.
- Auth: **none** (single-user design, see README s11). If an API key is added later it goes in an
  `Authorization: Bearer <key>` header and unauthenticated calls return `401`.

### Error shape [new]

All errors the gateway generates itself use:

```json
{ "error": { "code": "worker_unavailable", "message": "Human-readable text", "retryable": true } }
```

FastAPI's own validation errors (`422`) keep FastAPI's default shape.

| HTTP | `code` | Meaning | Retryable |
|---|---|---|---|
| 202 | n/a (not an error) | Worker was off; it is now starting. See s3.1 | after wait |
| 404 | `not_found` | Unknown conversation or job | no |
| 422 | n/a | Request body failed validation | no (fix request) |
| 503 | `worker_unavailable` | Worker failed to start, crashed, or is unreachable | yes |
| 503 | `worker_timeout` | Worker did not become healthy within the max wait (5 min) | yes |
| 401 | `unauthorized` | Only if auth is added later | no |

503 responses include a `Retry-After` header (seconds).

---

## 2. Endpoints that exist today

### `GET /health` [exists]

Gateway status plus a passthrough of the worker's health.

```json
{ "gateway": "ok", "worker": { "status": "ok" } }
```

`worker` is one of:
- the worker's own JSON (`{"status":"ok"}`)
- `"unreachable"` (connection refused / timeout)
- `{ "error": "non-JSON response (HTTP 404)" }` (something other than a worker answered)

Always returns `200`. After Phase 5 it also reports `worker_state` (see s3.2).

### `POST /conversations` [exists]

Creates an empty conversation.

Response `200`: `{ "conversation_id": 7 }`

### `GET /conversations/{conversation_id}` [exists]

Full stored history, oldest first (not just the sliding window).

Response `200`:
```json
{ "messages": [ { "role": "user", "content": "hi", "created_at": "..." },
                { "role": "assistant", "content": "Hello!", "created_at": "..." } ] }
```

Current behaviour: an unknown ID returns `200` with an empty list. **Target:** `404 not_found`. [new]

### `POST /chat` [exists]

Request:
```json
{ "message": "hi", "conversation_id": 7, "max_tokens": 256 }
```
- `message`: required string.
- `conversation_id`: optional. Omitted, `null`, or unknown means a new conversation is created.
- `max_tokens`: optional, default from `DEFAULT_MAX_TOKENS` (256).

Response `200`, `Content-Type: text/event-stream`:
- Header `X-Conversation-Id: <id>` (always set; the only place a newly created ID is returned).
- Body: Server-Sent Events passed through from the worker in OpenAI chunk format, one event per line pair:
  ```
  data: {"choices":[{"delta":{"content":"Hel"}}], ...}

  data: {"choices":[{"delta":{"content":"lo"}}], ...}

  data: [DONE]
  ```
  Clients read `choices[0].delta.content` and append. `data: [DONE]` ends the stream.
- Side effects: the user message is stored before streaming; the full assistant reply is stored once the
  stream ends. The model sees only the last `SLIDING_WINDOW_TURNS` stored messages.

---

## 3. New endpoints (Phase 3 and 5)

### 3.0 `GET /conversations` [new, Phase 3, task 3.6]

List conversations for the sidebar (task 3.12), newest activity first.

Response `200`:
```json
{ "conversations": [
    { "id": 7, "created_at": "...", "last_message_at": "...", "message_count": 12, "preview": "first 60 chars of first user message" }
] }
```

### 3.1 Cold start: `POST /chat` when the worker is off [new, Phase 5]

If the worker state is `IDLE`, `/chat` starts provisioning and returns immediately:

Response `202 Accepted`:
```json
{ "job_id": "j_8f3a2c", "state": "starting", "estimated_wait_seconds": 60, "conversation_id": 7 }
```
Also sets `X-Conversation-Id` and `Retry-After`.

If the state is already `STARTING`, the same `202` is returned with the **existing** job ID.
If the state is `READY`, `/chat` behaves as in s2 (streams `200`).

**What happens to the message on a 202? (Decided: Option A, see s5.)**
- **Option A (chosen):** the message is *not* stored or processed. The client waits for `ready`
  (s3.3 or s3.2) and **re-sends the same request**. The user sees one longer wait, no duplicates are possible, and the
  gateway needs no request queue for replay.
- **Option B:** the gateway stores the message and queues it, then auto-runs it when the worker is ready. This
  matches the original spec ("queue the request transparently") but then the client must fetch the result separately
  because the HTTP response already ended with 202.

The tracker's UI tasks (5.19: "auto-send once ready") assume Option A.

### 3.2 `GET /status/{job_id}` [new, Phase 5]

Response `200`:
```json
{ "job_id": "j_8f3a2c", "state": "starting", "elapsed_seconds": 22, "estimated_wait_seconds": 38, "error": null }
```
`state` is one of `starting`, `ready`, `failed`. When `failed`, `error` holds an error object (s1).
Unknown `job_id` returns `404`.

Also add `worker_state` (`idle` / `starting` / `ready`) to `/health`.

### 3.3 `GET /status/{job_id}/events` [new, Phase 5]

Server-Sent Events stream, alternative to polling. Sends the current state immediately, then on each change:

```
event: worker_starting
data: {"job_id":"j_8f3a2c","elapsed_seconds":5}

event: worker_ready
data: {"job_id":"j_8f3a2c"}

event: worker_failed
data: {"job_id":"j_8f3a2c","error":{"code":"worker_timeout","message":"...","retryable":true}}
```
The stream closes after `worker_ready` or `worker_failed`.

---

## 4. Failure behaviour [new, Phase 5]

| Situation | Response |
|---|---|
| Worker fails to launch or become healthy within 5 min | Job goes to `failed`; gateway reprovisions on the next `/chat`. Direct calls get `503 worker_timeout` |
| Provider API down | `503 worker_unavailable` with `Retry-After` |
| Worker crashes **before** the first token | `503 worker_unavailable` (status line not yet sent) |
| Worker crashes **mid-stream** | The `200` is already sent, so the stream ends with `data: {"error":{"code":"worker_unavailable","message":"...","retryable":true}}` and then `data: [DONE]` |

On any failure: conversation history is preserved. For a mid-stream crash the partial assistant text is **not**
stored. The user message was stored before streaming, so a retry should send the same request again
(**Decided:** the gateway deletes the stored user turn when the stream fails before any assistant text is saved,
so the retry does not duplicate it).

---

## 5. Decisions (agreed)

1. **Message on a 202: Option A.** The message is not stored or processed; the client waits for `ready` and re-sends.
   No replay queue is needed for a single-user app.
2. **Retry after failure:** if a stream fails before any assistant text is saved, the gateway deletes the stored user
   turn, so the client's retry (same message) never creates a duplicate.
3. **Auth:** none while the URL is an ephemeral Cloudflare tunnel. Add one static API key
   (`Authorization: Bearer <key>`, `401 unauthorized`) right before the gateway gets a permanent public URL
   (Phase 6, Fly.io/Railway). Tracker tasks 2.1/2.2/3.11 stay Skipped until then.
4. **Wait estimate:** fixed `estimated_wait_seconds` (start at 60, set via config). Switch to a rolling average of the
   last 5-10 real cold starts only after Phase 6 produces data.

## 6. Task map

| Section | Tracker tasks |
|---|---|
| 3.0 `GET /conversations` | 3.6, 3.12 |
| 3.1 / 3.2 / 3.3 | 5.11-5.14, UI 5.18-5.19 |
| 4 Failure behaviour | 5.16, 5.17, 5.20, 5.23 |
