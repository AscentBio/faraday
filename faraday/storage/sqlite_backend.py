from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from faraday.config import get_path_value
from faraday.storage.base import StorageBackend


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStorageBackend(StorageBackend):
    """Local SQLite backend for Faraday relational persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def backend_name(self) -> str:
        return "sqlite"

    @classmethod
    def from_env(cls) -> "SQLiteStorageBackend":
        default_path = str(Path.home() / ".faraday" / "faraday.db")
        db_path = get_path_value("storage", "sqlite_path", default=default_path) or default_path
        return cls(db_path=db_path)

    def initialize(self) -> None:
        with self._lock:
            if self._conn is not None:
                return

            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

            self._create_schema(conn)
            conn.commit()
            self._conn = conn

    def healthcheck(self) -> bool:
        try:
            self.initialize()
            assert self._conn is not None
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                chat_id TEXT,
                query_id TEXT UNIQUE,
                conversation_title TEXT,
                query_type TEXT,
                query_subtype TEXT,
                app_version REAL,
                project_label TEXT,
                model TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT,
                duration_seconds INTEGER,
                total_time REAL,
                num_steps INTEGER,
                step_stats_json TEXT,
                query_text TEXT,
                first_query_bool INTEGER,
                usage_cost_cumulative_dollars REAL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_user_project
                ON usage_records(user_id, project_label);

            CREATE TABLE IF NOT EXISTS user_balances (
                user_id TEXT PRIMARY KEY,
                user_balance_dollars REAL DEFAULT 0.0,
                usage_cost_cumulative_dollars REAL DEFAULT 0.0,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS tool_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                updated_at TEXT,
                user_id TEXT,
                chat_id TEXT,
                tool_run_id TEXT UNIQUE,
                tool_name TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL,
                tool_args_json TEXT,
                tool_results_json TEXT,
                error_message TEXT
            );
            """
        )
        # Lightweight migration for pre-existing local databases.
        try:
            conn.execute(
                "ALTER TABLE usage_records ADD COLUMN usage_cost_cumulative_dollars REAL"
            )
        except sqlite3.OperationalError:
            pass

    def _conn_or_raise(self) -> sqlite3.Connection:
        self.initialize()
        if self._conn is None:
            raise RuntimeError("SQLite connection not initialized")
        return self._conn

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value if value is not None else {})

    @staticmethod
    def _parse_json_field(row: dict[str, Any], key: str) -> None:
        raw = row.get(key)
        if isinstance(raw, str):
            try:
                row[key] = json.loads(raw)
            except Exception:
                pass

    # ── Usage records ────────────────────────────────────────────────────

    def create_usage_record(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._conn_or_raise()
            payload = dict(data)
            payload.setdefault("updated_at", _utc_now_iso())
            payload["step_stats_json"] = self._json_dumps(payload.get("step_stats"))
            payload["first_query_bool"] = int(bool(payload.get("first_query_bool"))) if "first_query_bool" in payload else None

            conn.execute(
                """
                INSERT INTO usage_records (
                    user_id, chat_id, query_id, conversation_title, query_type, query_subtype,
                    app_version, project_label, model, status, started_at, completed_at,
                    updated_at, duration_seconds, total_time, num_steps, step_stats_json,
                    query_text, first_query_bool, usage_cost_cumulative_dollars
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    conversation_title=excluded.conversation_title,
                    query_type=excluded.query_type,
                    query_subtype=excluded.query_subtype,
                    app_version=excluded.app_version,
                    project_label=excluded.project_label,
                    model=excluded.model,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    updated_at=excluded.updated_at,
                    num_steps=excluded.num_steps,
                    step_stats_json=excluded.step_stats_json,
                    query_text=excluded.query_text,
                    first_query_bool=excluded.first_query_bool,
                    usage_cost_cumulative_dollars=excluded.usage_cost_cumulative_dollars
                """,
                (
                    payload.get("user_id"),
                    payload.get("chat_id"),
                    payload.get("query_id"),
                    payload.get("conversation_title"),
                    payload.get("query_type"),
                    payload.get("query_subtype"),
                    payload.get("app_version"),
                    payload.get("project_label"),
                    payload.get("model"),
                    payload.get("status"),
                    payload.get("started_at"),
                    payload.get("completed_at"),
                    payload.get("updated_at"),
                    payload.get("duration_seconds"),
                    payload.get("total_time"),
                    payload.get("num_steps"),
                    payload.get("step_stats_json"),
                    payload.get("query_text"),
                    payload.get("first_query_bool"),
                    payload.get("usage_cost_cumulative_dollars"),
                ),
            )
            conn.commit()

            qid = payload.get("query_id")
            if not qid:
                return None
            row = conn.execute(
                "SELECT * FROM usage_records WHERE query_id = ? LIMIT 1", (qid,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            self._parse_json_field(result, "step_stats_json")
            return result

    def update_usage_record_by_query_id(
        self,
        query_id: str,
        status: str,
        additional_data: Optional[dict[str, Any]] = None,
        mode: str = "dev",
    ) -> bool:
        del mode
        updates = dict(additional_data or {})
        updates["status"] = status
        updates["updated_at"] = _utc_now_iso()
        if "step_stats" in updates:
            updates["step_stats_json"] = self._json_dumps(updates.pop("step_stats"))

        return self._update_usage_row(query_id, updates)

    def complete_usage_record_by_query_id(
        self,
        query_id: str,
        final_status: str = "completed",
        final_data: Optional[dict[str, Any]] = None,
        mode: str = "dev",
    ) -> bool:
        del mode
        updates = dict(final_data or {})
        updates["status"] = final_status
        updates["updated_at"] = _utc_now_iso()
        updates.setdefault("completed_at", _utc_now_iso())
        if "step_stats" in updates:
            updates["step_stats_json"] = self._json_dumps(updates.pop("step_stats"))

        return self._update_usage_row(query_id, updates)

    def _update_usage_row(self, query_id: str, updates: dict[str, Any]) -> bool:
        if not query_id:
            return False
        if not updates:
            return True

        allowed_columns = {
            "conversation_title",
            "query_type",
            "query_subtype",
            "app_version",
            "project_label",
            "model",
            "status",
            "started_at",
            "completed_at",
            "updated_at",
            "duration_seconds",
            "total_time",
            "num_steps",
            "step_stats_json",
            "query_text",
            "first_query_bool",
            "usage_cost_cumulative_dollars",
        }

        filtered = {k: v for k, v in updates.items() if k in allowed_columns}
        if not filtered:
            return True

        set_clause = ", ".join(f"{k} = ?" for k in filtered.keys())
        params = list(filtered.values()) + [query_id]

        with self._lock:
            conn = self._conn_or_raise()
            cur = conn.execute(
                f"UPDATE usage_records SET {set_clause} WHERE query_id = ?",
                params,
            )
            conn.commit()
            return cur.rowcount > 0

    def get_conversation_title(
        self,
        user_id: str,
        chat_id: str,
        mode: str = "dev",
    ) -> Optional[str]:
        del mode
        with self._lock:
            conn = self._conn_or_raise()
            row = conn.execute(
                """
                SELECT conversation_title
                FROM usage_records
                WHERE user_id = ? AND chat_id = ? AND conversation_title IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id, chat_id),
            ).fetchone()
            if not row:
                return None
            return row["conversation_title"]

    # ── User balance / cost tracking ─────────────────────────────────────

    def get_user_balance(self, user_id: str, mode: str = "dev") -> Optional[float]:
        del mode
        with self._lock:
            conn = self._conn_or_raise()
            row = conn.execute(
                "SELECT user_balance_dollars FROM user_balances WHERE user_id = ? LIMIT 1",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            value = row["user_balance_dollars"]
            return float(value) if value is not None else 0.0

    def set_user_balance(self, user_id: str, balance: float, mode: str = "dev") -> bool:
        del mode
        with self._lock:
            conn = self._conn_or_raise()
            conn.execute(
                """
                INSERT INTO user_balances (user_id, user_balance_dollars, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    user_balance_dollars=excluded.user_balance_dollars,
                    updated_at=excluded.updated_at
                """,
                (user_id, float(balance), _utc_now_iso()),
            )
            conn.commit()
            return True

    def get_usage_cost_cumulative(self, user_id: str, mode: str = "dev") -> float:
        del mode
        with self._lock:
            conn = self._conn_or_raise()
            row = conn.execute(
                """
                SELECT usage_cost_cumulative_dollars
                FROM user_balances
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if not row:
                return 0.0
            value = row["usage_cost_cumulative_dollars"]
            return float(value) if value is not None else 0.0

    def set_usage_cost_cumulative(self, user_id: str, value: float, mode: str = "dev") -> bool:
        del mode
        with self._lock:
            conn = self._conn_or_raise()
            conn.execute(
                """
                INSERT INTO user_balances (user_id, usage_cost_cumulative_dollars, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    usage_cost_cumulative_dollars=excluded.usage_cost_cumulative_dollars,
                    updated_at=excluded.updated_at
                """,
                (user_id, float(value), _utc_now_iso()),
            )
            conn.commit()
            return True

    # ── Tool jobs ────────────────────────────────────────────────────────

    def create_tool_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        payload = dict(job_data)
        payload.setdefault("created_at", _utc_now_iso())
        payload.setdefault("updated_at", payload["created_at"])
        payload.setdefault("status", "pending")

        with self._lock:
            conn = self._conn_or_raise()
            conn.execute(
                """
                INSERT INTO tool_jobs (
                    created_at, updated_at, user_id, chat_id, tool_run_id, tool_name,
                    status, started_at, completed_at, duration_seconds, tool_args_json,
                    tool_results_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_run_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    duration_seconds=excluded.duration_seconds,
                    tool_args_json=excluded.tool_args_json,
                    tool_results_json=excluded.tool_results_json,
                    error_message=excluded.error_message
                """,
                (
                    payload.get("created_at"),
                    payload.get("updated_at"),
                    payload.get("user_id"),
                    payload.get("chat_id"),
                    payload.get("tool_run_id"),
                    payload.get("tool_name"),
                    payload.get("status"),
                    payload.get("started_at"),
                    payload.get("completed_at"),
                    payload.get("duration_seconds"),
                    self._json_dumps(payload.get("tool_args")),
                    self._json_dumps(payload.get("tool_results")),
                    payload.get("error_message"),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM tool_jobs WHERE tool_run_id = ? LIMIT 1",
                (payload.get("tool_run_id"),),
            ).fetchone()
            return self._tool_job_row_to_dict(row) if row else payload

    def update_tool_job(self, tool_run_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not tool_run_id:
            return None

        with self._lock:
            conn = self._conn_or_raise()
            existing = conn.execute(
                "SELECT * FROM tool_jobs WHERE tool_run_id = ? LIMIT 1",
                (tool_run_id,),
            ).fetchone()
            if not existing:
                return None

            payload = dict(existing)
            payload.update(updates or {})
            payload["tool_run_id"] = tool_run_id
            payload["updated_at"] = _utc_now_iso()

            if "tool_args_json" in payload and "tool_args" not in payload:
                self._parse_json_field(payload, "tool_args_json")
                payload["tool_args"] = payload.get("tool_args_json")
            if "tool_results_json" in payload and "tool_results" not in payload:
                self._parse_json_field(payload, "tool_results_json")
                payload["tool_results"] = payload.get("tool_results_json")

            self.create_tool_job(payload)
            row = conn.execute(
                "SELECT * FROM tool_jobs WHERE tool_run_id = ? LIMIT 1",
                (tool_run_id,),
            ).fetchone()
            return self._tool_job_row_to_dict(row) if row else None

    def get_tool_job(self, tool_run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._conn_or_raise()
            row = conn.execute(
                "SELECT * FROM tool_jobs WHERE tool_run_id = ? LIMIT 1",
                (tool_run_id,),
            ).fetchone()
            if not row:
                return None
            return self._tool_job_row_to_dict(row)

    def _tool_job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        self._parse_json_field(data, "tool_args_json")
        self._parse_json_field(data, "tool_results_json")
        data["tool_args"] = data.pop("tool_args_json", {})
        data["tool_results"] = data.pop("tool_results_json", {})
        return data
