"""``get_daily_totals`` — the authoritative answer to "how am I doing today?".

Totals are read from the ``daily_totals`` SQL view, which aggregates the meal
rows on every read. There is no counter to keep in sync, so an edit or a delete
is reflected the instant it lands. This is why corrections cannot double-count:
the number the user hears is always ``SUM()`` over the rows that currently
exist, never an incrementally maintained tally that could drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from src.db.queries import daily_totals
from src.utils.config import local_date_str
from src.utils.user_context import get_user_id


@tool("get_daily_totals")
def get_daily_totals(period: str = "today", date: Optional[str] = None) -> Dict[str, Any]:
    """Get calories and macros (grams) totalled for one day.

    Use this for any question about how the day is going — never estimate totals
    from the conversation, the numbers here are the real ones.

    Args:
        period: "today" or "yesterday". Ignored when `date` is given.
        date: Specific local date as YYYY-MM-DD.
    """
    if date:
        day = date
    elif (period or "today").lower().strip() == "yesterday":
        day = local_date_str(-1)
    else:
        day = local_date_str()

    # Deliberately the same key names log_meal returns in its `daily_totals`
    # block. One shape for one concept - the model never has to learn two.
    return daily_totals(get_user_id(), day)
