"""Point every test at throwaway databases before ``src`` is imported."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="calorai-tests-"))
os.environ["CALORAI_DB"] = str(_TMP / "test.db")
os.environ["CALORAI_CHECKPOINT_DB"] = str(_TMP / "ckpt.db")
os.environ["CALORAI_LATENCY_LOG"] = str(_TMP / "latency.jsonl")
os.environ["CALORAI_MOCK"] = "1"
os.environ["USDA_API_KEY"] = ""  # unit tests never touch the network

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.db.schema import reset_database  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db() -> None:
    """Every test starts from empty tables."""
    reset_database()
