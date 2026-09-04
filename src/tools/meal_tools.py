"""Meal CRUD tools: log and retrieve.

``log_meal`` deliberately returns the day's totals alongside the new row. The
agent almost always wants to confirm "that's 520 cal, you're at 1,340 today", and
folding that into the write saves a second tool round-trip on the most common
path in the whole product.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.db.queries import daily_totals, insert_meal, list_meals
from src.utils.config import local_date_str
from src.utils.user_context import get_user_id

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def _summarize(meal: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a meal row to the fields the model needs (keeps context small)."""
    return {
        "meal_id": meal["id"],
        "meal_name": meal["meal_name"],
        "description": meal["description"],
        "meal_type": meal["meal_type"],
        "date": meal["meal_date"],
        "time": meal["timestamp"][11:16],
        "calories": meal["calories"],
        "protein": meal["protein"],
        "carbs": meal["carbs"],
        "fat": meal["fat"],
    }


@tool("log_meal")
def log_meal(
    meal_name: str,
    calories: float,
    protein: float = 0.0,
    carbs: float = 0.0,
    fat: float = 0.0,
    description: str = "",
    meal_type: str = "snack",
    source: str = "text",
    meal_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a NEW meal the user just ate. Call lookup_nutrition first for the macros.

    Do NOT use this to fix a meal that is already logged — that is update_meal.
    Combine everything the user described as one eating occasion into a single
    call; a photo plus a caption about it is ONE meal, not two.

    Args:
        meal_name: Short label, e.g. "2 parathas and chai".
        calories: Total kcal for the whole meal.
        protein: Total protein in grams.
        carbs: Total carbohydrate in grams.
        fat: Total fat in grams.
        description: Portion detail worth remembering, e.g. "~2/3 of the box".
        meal_type: One of breakfast, lunch, dinner, snack.
        source: "text", "vision", or "vision+text" if a photo was involved.
        meal_date: Local YYYY-MM-DD. Omit for today; use it when re-logging an
            earlier day's meals.
    """
    meal_type = meal_type.lower().strip()
    if meal_type not in MEAL_TYPES:
        meal_type = "snack"

    meal = insert_meal(
        user_id=get_user_id(),
        meal_name=meal_name,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        description=description,
        meal_type=meal_type,
        source=source,
    )
    # meal_date is applied post-insert so the stored UTC timestamp stays truthful
    # about when it was recorded while the day it counts toward can be steered.
    if meal_date and meal_date != meal["meal_date"]:
        from src.db.schema import get_connection

        conn = get_connection()
        conn.execute("UPDATE meals SET meal_date = ? WHERE id = ?", (meal_date, meal["id"]))
        conn.commit()
        meal["meal_date"] = meal_date

    return {
        "logged": _summarize(meal),
        "daily_totals": daily_totals(get_user_id(), meal["meal_date"]),
    }


@tool("get_meals")
def get_meals(
    period: str = "today",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name_contains: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Retrieve previously logged meals, newest first.

    Use this to answer "what did I eat", to find the meal a correction refers to
    (pass name_contains, e.g. "roti"), and to replay a past day for requests like
    "same as yesterday".

    Args:
        period: "today", "yesterday", "week" (last 7 days), or "all".
        start_date: Local YYYY-MM-DD; overrides period when given.
        end_date: Local YYYY-MM-DD; overrides period when given.
        name_contains: Only meals whose name or description contains this text.
        limit: Max rows to return.
    """
    if start_date or end_date:
        window_start, window_end = start_date, end_date
    else:
        period = (period or "today").lower().strip()
        if period == "yesterday":
            window_start = window_end = local_date_str(-1)
        elif period == "week":
            window_start, window_end = local_date_str(-6), local_date_str()
        elif period == "all":
            window_start = window_end = None
        else:
            window_start = window_end = local_date_str()

    meals: List[Dict[str, Any]] = list_meals(
        user_id=get_user_id(),
        start_date=window_start,
        end_date=window_end,
        name_contains=name_contains,
        limit=limit,
    )
    return {
        "period": period,
        "start_date": window_start,
        "end_date": window_end,
        "count": len(meals),
        "meals": [_summarize(m) for m in meals],
    }
