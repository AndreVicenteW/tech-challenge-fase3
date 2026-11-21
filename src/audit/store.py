import json
import sqlite3

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.database.connection import PROJECT_ROOT


AUDIT_DB = PROJECT_ROOT / "data" / "local" / "audit.db"


class AuditStore:
    def __init__(self, db_path: Path = AUDIT_DB):
        self.db_path = Path(db_path)

    def save(self, event: dict) -> None:
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    patient_id TEXT,
                    status TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    validation_json TEXT,
                    draft TEXT,
                    proposed_answer TEXT,
                    answer TEXT,
                    message TEXT
                )
            """)

            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO audit_events (
                        request_id,
                        created_at,
                        question,
                        patient_id,
                        status,
                        sources_json,
                        validation_json,
                        draft,
                        proposed_answer,
                        answer,
                        message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["request_id"],
                        datetime.now(timezone.utc).isoformat(),
                        event["question"],
                        event.get("patient_id"),
                        event["status"],
                        json.dumps(
                            event.get("sources", []),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            event.get("validation"),
                            ensure_ascii=False,
                        ),
                        event.get("draft"),
                        event.get("proposed_answer"),
                        event.get("answer"),
                        event.get("message"),
                    ),
                )

    def get(self, request_id: str) -> dict | None:
        if not self.db_path.exists():
            return None

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                """
                SELECT *
                FROM audit_events
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["sources"] = json.loads(
            result.pop("sources_json")
        )
        result["validation"] = json.loads(
            result.pop("validation_json")
        )

        return result