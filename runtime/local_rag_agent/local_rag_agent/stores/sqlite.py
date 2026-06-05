from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteRunStore:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    @classmethod
    def in_memory(cls) -> "SQLiteRunStore":
        return cls(sqlite3.connect(":memory:"))

    @classmethod
    def from_path(cls, path: Path) -> "SQLiteRunStore":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path))

    def create_run(
        self,
        run_id: str,
        thread_id: str = "",
        intent: str = "",
        workflow: str = "",
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        self.connection.execute(
            """
            INSERT INTO runs (run_id, thread_id, intent, workflow, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, thread_id, intent, workflow, status, now, now, _to_json(metadata or {})),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            """
            SELECT run_id, thread_id, intent, workflow, status, metadata_json
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "thread_id": row["thread_id"] or "",
            "intent": row["intent"] or "",
            "workflow": row["workflow"] or "",
            "status": row["status"],
            "metadata": _from_json(row["metadata_json"]),
        }

    def write_checkpoint(
        self,
        run_id: str,
        node_id: str,
        state: dict[str, Any],
        trace: dict[str, Any] | None = None,
    ) -> str:
        checkpoint_id = uuid.uuid4().hex
        self.connection.execute(
            """
            INSERT INTO checkpoints (checkpoint_id, run_id, node_id, state_json, trace_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (checkpoint_id, run_id, node_id, _to_json(state), _to_json(trace or {}), _now()),
        )
        self.connection.commit()
        return checkpoint_id

    def list_checkpoints(self, run_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT checkpoint_id, run_id, node_id, state_json, trace_json
            FROM checkpoints
            WHERE run_id = ?
            ORDER BY created_at, checkpoint_id
            """,
            (run_id,),
        ).fetchall()
        return [
            {
                "checkpoint_id": row["checkpoint_id"],
                "run_id": row["run_id"],
                "node_id": row["node_id"],
                "state": _from_json(row["state_json"]),
                "trace": _from_json(row["trace_json"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        self.connection.executescript(schema_path.read_text(encoding="utf-8"))
        self.connection.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _from_json(raw: str) -> dict[str, Any]:
    payload = json.loads(raw or "{}")
    return payload if isinstance(payload, dict) else {}
