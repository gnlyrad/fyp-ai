"""Small durable job store used locally in place of the production PostgreSQL queue."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobStore:
    """Persist queued ingestion/analysis jobs in SQLite for local development.

    The repository interface deliberately contains no SQLite-specific details in
    callers, so it can be backed by PostgreSQL and a real queue in deployment.
    """

    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, source_type: str, payload: dict[str, Any]) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, "queued", source_type, json.dumps(payload), None, None, now, now),
            )
        return job_id

    def update(self, job_id: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status=?, result_json=?, error_message=?, updated_at=? WHERE job_id=?",
                (status, json.dumps(result) if result is not None else None, error, now, job_id),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(payload.pop("payload_json"))
        payload["result"] = json.loads(payload.pop("result_json")) if payload["result_json"] else None
        return payload
