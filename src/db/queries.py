"""Data access layer. Every SQL statement in the project lives here.

Tools call these functions; they never open a cursor themselves. That keeps the
tool layer about *policy* (what to log, when to ask) and this layer about
*storage*, and it makes the correction path easy to reason about: an update is a
single ``UPDATE ... WHERE id = ?``, never a delete-plus-insert.

Every function takes the connection through :func:`src.db.schema.connection`,
which holds a re-entrant lock for the duration of the block. Parallel tool calls
run on real threads, so a multi-statement sequence (insert, then read back the
new row) has to be atomic or another thread can interleave between the two.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from src.db.schema import connection
from src.utils.config import local_date_str, to_local_date, utc_now_iso

MACRO_FIELDS = ("calories", "protein", "carbs", "fat")
_EDITABLE = ("meal_name", "description", "meal_type", *MACRO_FIELDS)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _fetch_meal(conn: sqlite3.Connection, user_id: str, meal_id: int) -> Optional[Dict[str, Any]]:
    """Read one meal using an already-held connection."""
    row = conn.execute(
        "SELECT * FROM meals WHERE id = ? AND user_id = ?", (meal_id, user_id)
    ).fetchone()
    return _row_to_dict(row) if row else None


# --- meals -------------------------------------------------------------------


def insert_meal(
    user_id: str,
    meal_name: str,
    calories: float,
    protein: float = 0.0,
    carbs: float = 0.0,
    fat: float = 0.0,
    description: str = "",
    meal_type: str = "snack",
    source: str = "text",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert one meal row and return it.

    The insert and the read-back share one locked block; splitting them let a
    concurrent writer interleave and made this return None intermittently.
    """
    now = utc_now_iso()
    ts = timestamp or now
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO meals (user_id, meal_name, description, calories, protein,
                               carbs, fat, timestamp, meal_date, meal_type, source,
                               created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                meal_name,
                description,
                round(float(calories), 1),
                round(float(protein), 1),
                round(float(carbs), 1),
                round(float(fat), 1),
                ts,
                to_local_date(ts),
                meal_type,
                source,
                now,
                now,
            ),
        )
        conn.commit()
        meal = _fetch_meal(conn, user_id, int(cur.lastrowid))
    if meal is None:  # pragma: no cover - would mean the insert silently failed
        raise RuntimeError(f"meal insert for {user_id!r} did not persist")
    return meal


def get_meal(user_id: str, meal_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single meal, scoped to its owner."""
    with connection() as conn:
        return _fetch_meal(conn, user_id, meal_id)


def list_meals(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name_contains: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query meals by local-date range and/or name fragment, newest first.

    ``name_contains`` is what makes corrections work without a dedicated search
    tool: "that was 3 rotis not 2" resolves to the most recent roti row.
    """
    sql = "SELECT * FROM meals WHERE user_id = ?"
    params: List[Any] = [user_id]
    if start_date:
        sql += " AND meal_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND meal_date <= ?"
        params.append(end_date)
    if name_contains:
        sql += " AND (LOWER(meal_name) LIKE ? OR LOWER(description) LIKE ?)"
        pattern = f"%{name_contains.lower()}%"
        params.extend([pattern, pattern])
    sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    params.append(int(limit))
    with connection() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def update_meal_row(user_id: str, meal_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    """Update an existing meal in place. Unknown/None fields are ignored.

    Returns the updated row, or ``None`` if the meal does not exist for the user.
    """
    updates = {k: v for k, v in fields.items() if k in _EDITABLE and v is not None}
    for macro in MACRO_FIELDS:
        if macro in updates:
            updates[macro] = round(float(updates[macro]), 1)

    with connection() as conn:
        if _fetch_meal(conn, user_id, meal_id) is None:
            return None
        if updates:
            assignments = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE meals SET {assignments}, updated_at = ? WHERE id = ? AND user_id = ?",
                (*updates.values(), utc_now_iso(), meal_id, user_id),
            )
            conn.commit()
        return _fetch_meal(conn, user_id, meal_id)


def delete_meal_row(user_id: str, meal_id: int) -> bool:
    """Delete a meal. Returns True if a row was actually removed."""
    with connection() as conn:
        cur = conn.execute(
            "DELETE FROM meals WHERE id = ? AND user_id = ?", (meal_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0


def daily_totals(user_id: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Read totals for one local day straight off the ``daily_totals`` view."""
    day = date or local_date_str()
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM daily_totals WHERE user_id = ? AND meal_date = ?", (user_id, day)
        ).fetchone()
    if row is None:
        return {
            "date": day,
            "meal_count": 0,
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
        }
    return {
        "date": day,
        "meal_count": row["meal_count"],
        "calories": row["calories"],
        "protein": row["protein"],
        "carbs": row["carbs"],
        "fat": row["fat"],
    }


# --- memories ----------------------------------------------------------------


def upsert_memory(
    user_id: str, key: str, value: str, category: str = "fact"
) -> Dict[str, Any]:
    """Write or overwrite a memory. Keyed on ``(user_id, key)`` so restating a
    preference updates it rather than accumulating duplicates."""
    now = utc_now_iso()
    normalized = key.strip().lower()
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO memories (user_id, key, value, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = excluded.updated_at
            """,
            (user_id, normalized, value, category, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM memories WHERE user_id = ? AND key = ?", (user_id, normalized)
        ).fetchone()
    return _row_to_dict(row)


def list_memories(
    user_id: str, categories: Optional[List[str]] = None, limit: int = 100
) -> List[Dict[str, Any]]:
    """List memories, most recently updated first, optionally by category."""
    sql = "SELECT * FROM memories WHERE user_id = ?"
    params: List[Any] = [user_id]
    if categories:
        sql += f" AND category IN ({','.join('?' * len(categories))})"
        params.extend(categories)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(limit))
    with connection() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def touch_memories(user_id: str, keys: List[str]) -> None:
    """Record that these memories were surfaced - feeds recall ranking."""
    if not keys:
        return
    with connection() as conn:
        conn.executemany(
            "UPDATE memories SET last_used = ?, use_count = use_count + 1 "
            "WHERE user_id = ? AND key = ?",
            [(utc_now_iso(), user_id, k) for k in keys],
        )
        conn.commit()


def delete_memory(user_id: str, key: str) -> bool:
    """Remove a memory by key."""
    with connection() as conn:
        cur = conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key.strip().lower()),
        )
        conn.commit()
        return cur.rowcount > 0


# --- nutrition cache ---------------------------------------------------------


def get_cached_nutrition(food_key: str) -> Optional[Dict[str, Any]]:
    """Look up a memoised nutrition entry."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM nutrition_cache WHERE food_key = ?", (food_key,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def put_cached_nutrition(
    food_key: str,
    food_name: str,
    quantity: str,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    source: str,
) -> None:
    """Store a nutrition entry so the next identical lookup costs no tokens."""
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO nutrition_cache (food_key, food_name, quantity, calories,
                                         protein, carbs, fat, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (food_key) DO UPDATE SET
                calories = excluded.calories, protein = excluded.protein,
                carbs = excluded.carbs, fat = excluded.fat, source = excluded.source
            """,
            (
                food_key,
                food_name,
                quantity,
                round(float(calories), 1),
                round(float(protein), 1),
                round(float(carbs), 1),
                round(float(fat), 1),
                source,
                utc_now_iso(),
            ),
        )
        conn.commit()
