"""SQLite schema and connection handling.

Three tables and one view:

* ``meals``            - one row per logged meal, corrections update in place.
* ``memories``         - durable facts about the user, keyed and upsertable.
* ``nutrition_cache``  - memoised nutrition lookups (survives restarts).
* ``daily_totals``     - view that aggregates ``meals`` per user per local day.

The view is the single source of truth for "how am I doing today" so totals can
never drift from the meal rows that produced them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from src.utils.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    meal_name   TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    calories    REAL    NOT NULL DEFAULT 0,
    protein     REAL    NOT NULL DEFAULT 0,
    carbs       REAL    NOT NULL DEFAULT 0,
    fat         REAL    NOT NULL DEFAULT 0,
    timestamp   TEXT    NOT NULL,                     -- UTC ISO-8601
    meal_date   TEXT    NOT NULL,                     -- local YYYY-MM-DD
    meal_type   TEXT    NOT NULL DEFAULT 'snack',     -- breakfast|lunch|dinner|snack
    source      TEXT    NOT NULL DEFAULT 'text',      -- text|vision|vision+text
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals (user_id, meal_date);
CREATE INDEX IF NOT EXISTS idx_meals_user_ts   ON meals (user_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    category   TEXT NOT NULL DEFAULT 'fact',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used  TEXT,
    use_count  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_memories_user_cat ON memories (user_id, category);

CREATE TABLE IF NOT EXISTS nutrition_cache (
    food_key   TEXT PRIMARY KEY,          -- normalised "food@quantity"
    food_name  TEXT NOT NULL,
    quantity   TEXT NOT NULL DEFAULT '1 serving',
    calories   REAL NOT NULL,
    protein    REAL NOT NULL,
    carbs      REAL NOT NULL,
    fat        REAL NOT NULL,
    source     TEXT NOT NULL,             -- seed|llm
    created_at TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS daily_totals AS
SELECT
    user_id,
    meal_date,
    COUNT(*)                     AS meal_count,
    ROUND(SUM(calories), 1)      AS calories,
    ROUND(SUM(protein), 1)       AS protein,
    ROUND(SUM(carbs), 1)         AS carbs,
    ROUND(SUM(fat), 1)           AS fat
FROM meals
GROUP BY user_id, meal_date;
"""

_connection: Optional[sqlite3.Connection] = None


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return the process-wide connection, creating the schema on first use."""
    global _connection
    if _connection is None:
        path = db_path or settings.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: LangGraph may execute tool nodes on a worker
        # thread. Writes are short and serialised by SQLite's own locking.
        _connection = sqlite3.connect(str(path), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
        _connection.executescript(SCHEMA_SQL)
        _connection.commit()
    return _connection


def close_connection() -> None:
    """Close the shared connection (tests reset between databases)."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def reset_database(db_path: Optional[Path] = None) -> None:
    """Drop every row. Used by the eval runner to start from a clean slate."""
    conn = get_connection(db_path)
    conn.executescript(
        "DELETE FROM meals; DELETE FROM memories; DELETE FROM nutrition_cache;"
    )
    conn.commit()
