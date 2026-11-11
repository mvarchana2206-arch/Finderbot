"""
storage.py
==========
SQLite database manager and status constants for FinderBot.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Item status constants
STATUS_FOUND = "Found"
STATUS_UNDER_VERIFICATION = "Under Verification"
STATUS_CLAIM_APPROVED = "Claim Approved"
STATUS_RELEASED = "Released"
STATUS_REJECTED = "Rejected"


class DatabaseManager:
    def __init__(self, db_path: str = "data/finderbot.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Items table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    category TEXT,
                    item_type TEXT,
                    primary_color TEXT,
                    secondary_color TEXT,
                    material TEXT,
                    physical_traits TEXT,
                    distinctive_marks TEXT,
                    verification_notes TEXT,
                    dominant_colors TEXT,
                    image_quality_notes TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    found_location TEXT,
                    image_data BLOB,
                    status TEXT,
                    registered_at TEXT
                )
                """
            )
            # Evaluations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id TEXT PRIMARY KEY,
                    item_id TEXT,
                    match_score INTEGER,
                    confidence_tier TEXT,
                    matched_aspects TEXT,
                    discrepancies TEXT,
                    recommended_action TEXT,
                    score_breakdown TEXT,
                    raw_result TEXT,
                    latency_seconds REAL,
                    evaluated_at TEXT,
                    FOREIGN KEY(item_id) REFERENCES items(item_id)
                )
                """
            )
            # Audit log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT,
                    admin_action TEXT,
                    previous_status TEXT,
                    new_status TEXT,
                    performed_by TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.commit()

    def register_item(
        self,
        category: str,
        item_type: str,
        primary_color: str,
        secondary_color: str,
        material: str,
        physical_traits: List[str],
        distinctive_marks: List[str],
        verification_notes: str,
        dominant_colors: List[str],
        image_quality_notes: str,
        image_width: int,
        image_height: int,
        found_location: str,
        image_data: Optional[bytes] = None,
    ) -> str:
        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO items (
                    item_id, category, item_type, primary_color, secondary_color,
                    material, physical_traits, distinctive_marks, verification_notes,
                    dominant_colors, image_quality_notes, image_width, image_height,
                    found_location, image_data, status, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    category,
                    item_type,
                    primary_color,
                    secondary_color,
                    material,
                    json.dumps(physical_traits),
                    json.dumps(distinctive_marks),
                    verification_notes,
                    json.dumps(dominant_colors),
                    image_quality_notes,
                    image_width,
                    image_height,
                    found_location,
                    image_data,
                    STATUS_FOUND,
                    now,
                ),
            )
            cursor.execute(
                """
                INSERT INTO audit_log (item_id, admin_action, previous_status, new_status, performed_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, "Item Registered", None, STATUS_FOUND, "system", now),
            )
            conn.commit()
        return item_id

    def list_items(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM items ORDER BY registered_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_item_image(self, item_id: str) -> Optional[bytes]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_data FROM items WHERE item_id = ?", (item_id,))
            row = cursor.fetchone()
            return row["image_data"] if row and row["image_data"] else None

    def update_item_status(self, item_id: str, new_status: str, action_desc: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM items WHERE item_id = ?", (item_id,))
            row = cursor.fetchone()
            prev_status = row["status"] if row else None

            cursor.execute("UPDATE items SET status = ? WHERE item_id = ?", (new_status, item_id))
            cursor.execute(
                """
                INSERT INTO audit_log (item_id, admin_action, previous_status, new_status, performed_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, action_desc or f"Status changed to {new_status}", prev_status, new_status, "admin", now),
            )
            conn.commit()

    def save_evaluation(
        self,
        item_id: str,
        match_score: int,
        confidence_tier: str,
        matched_aspects: List[str],
        discrepancies: List[str],
        recommended_action: str,
        score_breakdown: Dict[str, Any],
        raw_result: Dict[str, Any],
        latency_seconds: float,
    ) -> str:
        eval_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO evaluations (
                    eval_id, item_id, match_score, confidence_tier, matched_aspects,
                    discrepancies, recommended_action, score_breakdown, raw_result,
                    latency_seconds, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    item_id,
                    match_score,
                    confidence_tier,
                    json.dumps(matched_aspects),
                    json.dumps(discrepancies),
                    recommended_action,
                    json.dumps(score_breakdown),
                    json.dumps(raw_result),
                    latency_seconds,
                    now,
                ),
            )
            conn.commit()
        return eval_id

    def list_all_evaluations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluations ORDER BY evaluated_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def count_items(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def count_evaluations(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]

    def count_items_by_status(self, status: str) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM items WHERE status = ?", (status,)).fetchone()[0]

    def avg_match_score(self) -> float:
        with self._get_connection() as conn:
            res = conn.execute("SELECT AVG(match_score) FROM evaluations").fetchone()[0]
            return float(res) if res is not None else 0.0

    def avg_latency(self) -> float:
        with self._get_connection() as conn:
            res = conn.execute("SELECT AVG(latency_seconds) FROM evaluations").fetchone()[0]
            return float(res) if res is not None else 0.0

    def match_score_distribution(self) -> List[int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT match_score FROM evaluations")
            return [row[0] for row in cursor.fetchall()]

    def confidence_tier_counts(self) -> Dict[str, int]:
        counts = {"High": 0, "Probable": 0, "Unlikely": 0}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT confidence_tier, COUNT(*) FROM evaluations GROUP BY confidence_tier")
            for row in cursor.fetchall():
                if row[0] in counts:
                    counts[row[0]] = row[1]
        return counts

    def get_audit_log(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]