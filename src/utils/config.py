"""Central configuration, loaded once from the environment.

Everything the agent can be re-pointed at (models, database path, timezone)
lives here so no module reads ``os.environ`` on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:  # python-dotenv is a convenience, not a hard requirement at import time.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    text_model: str
    vision_model: str
    nutrition_model: str
    db_path: Path
    checkpoint_db_path: Path
    latency_log: Path
    tz_offset_hours: float
    max_memories: int
    max_history_tokens: int
    mock: bool
    usda_api_key: str

    @property
    def tz(self) -> timezone:
        """The user's timezone, used to decide which day a meal belongs to."""
        return timezone(timedelta(hours=self.tz_offset_hours))


def _load() -> Settings:
    return Settings(
        text_model=_env("TEXT_MODEL", "claude-haiku-4-5"),
        vision_model=_env("VISION_MODEL", "claude-sonnet-5"),
        nutrition_model=_env("NUTRITION_MODEL", "claude-haiku-4-5"),
        db_path=_resolve(_env("CALORAI_DB", "data/calorai.db")),
        checkpoint_db_path=_resolve(_env("CALORAI_CHECKPOINT_DB", "data/checkpoints.db")),
        latency_log=_resolve(_env("CALORAI_LATENCY_LOG", "data/latency.jsonl")),
        tz_offset_hours=float(_env("CALORAI_TZ_OFFSET_HOURS", "5.5")),
        max_memories=int(_env("CALORAI_MAX_MEMORIES", "8")),
        # Recent-history budget the model sees per call; the full thread stays
        # in the checkpointer. Facts are in SQLite, so this can be small.
        max_history_tokens=int(_env("CALORAI_MAX_HISTORY_TOKENS", "2000")),
        mock=_env("CALORAI_MOCK", "0") not in ("0", "false", "False"),
        # Off unless set. DEMO_KEY works but is capped at 10 requests/hour.
        usda_api_key=os.environ.get("USDA_API_KEY", "").strip(),
    )


settings = _load()


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests that flip ``CALORAI_MOCK``."""
    global settings
    settings = _load()
    return settings


# --- time helpers ------------------------------------------------------------
# A "day" is the user's local calendar day, not UTC. Meals are stored with a UTC
# timestamp plus a precomputed local ``meal_date`` so daily totals are a cheap
# indexed lookup instead of a per-row timezone conversion.


def utc_now_iso() -> str:
    """Current instant as a UTC ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_now() -> datetime:
    """Current time in the user's local timezone."""
    return datetime.now(settings.tz)


def local_date_str(offset_days: int = 0) -> str:
    """Local calendar date as ``YYYY-MM-DD``; ``offset_days=-1`` is yesterday."""
    return (local_now() + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def to_local_date(utc_iso: str) -> str:
    """Map a stored UTC timestamp onto the user's local calendar date."""
    return datetime.fromisoformat(utc_iso).astimezone(settings.tz).strftime("%Y-%m-%d")
