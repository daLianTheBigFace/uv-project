import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "agent_traces.sqlite3"
TRACE_DB_PATH = Path(os.getenv("AGENT_TRACE_DB_PATH", str(_DEFAULT_DB_PATH))).expanduser()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_json_dumps(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"unserializable": str(payload)}, ensure_ascii=False)


def _connect() -> sqlite3.Connection:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TRACE_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_trace_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                mode TEXT NOT NULL,
                model_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                input_messages_json TEXT,
                final_output TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                tool_name TEXT,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_events_run_seq
            ON agent_events(run_id, seq)
            """
        )
        _ensure_agent_runs_columns(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation_id
            ON agent_runs(conversation_id)
            """
        )


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _ensure_agent_runs_columns(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "agent_runs", "conversation_id"):
        conn.execute("ALTER TABLE agent_runs ADD COLUMN conversation_id TEXT")


def start_run(mode: str, model_name: str, input_messages: Any, conversation_id: str | None = None) -> str:
    init_trace_db()
    run_id = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (run_id, conversation_id, mode, model_name, started_at, input_messages_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, conversation_id, mode, model_name, _utc_now_iso(), _safe_json_dumps(input_messages)),
        )
    return run_id


def log_event(run_id: str, seq: int, event_type: str, payload: Any, tool_name: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_events (run_id, seq, event_type, tool_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, seq, event_type, tool_name, _safe_json_dumps(payload), _utc_now_iso()),
        )


def finish_run(run_id: str, final_output: str | None = None, error: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE agent_runs
            SET ended_at = ?, final_output = ?, error = ?
            WHERE run_id = ?
            """,
            (_utc_now_iso(), final_output, error, run_id),
        )

