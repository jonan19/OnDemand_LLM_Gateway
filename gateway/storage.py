import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "gateway.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_conversation() -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO conversations (created_at) VALUES (?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conversation_id = cur.lastrowid
    conn.close()
    return conversation_id


def conversation_exists(conversation_id: int) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return row is not None


def add_message(conversation_id: int, role: str, content: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_messages(conversation_id: int, limit: int = 10) -> list[dict]:
    """Sliding-window context: the last `limit` turns, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_full_history(conversation_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT role, content, created_at FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        for r in rows
    ]
