"""Meal CRUD tools: log and retrieve.

``log_meal`` deliberately returns the day's totals alongside the new row. The
agent almost always wants to confirm "that's 520 cal, you're at 1,340 today", and
folding that into the write saves a second tool round-trip on the most common
path in the whole product.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.db.queries import (
    daily_totals,
    delete_meal_row,
    get_meal,
    insert_meal,
    list_meals,
    update_meal_row,
)
from src.utils.config import local_date_str
from src.utils.user_context import get_user_id

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")

# Sanity bounds on model-supplied numbers. A very large single meal is ~3000
# kcal, so anything past 6000 is a typo or a parsing slip ("99999 rotis"), not
# food. These are rejected rather than clamped: a clamped value is a wrong
# number stored silently, whereas an error lets the agent ask what was meant.
MAX_MEAL_CALORIES = 6000.0
MAX_MACRO_GRAMS = 1500.0


def _validate_macros(
    calories: Optional[float],
    protein: Optional[float],
    carbs: Optional[float],
    fat: Optional[float],
) -> Optional[str]:
    """Return an error message if the numbers are not plausible food, else None."""
    checks = (
        ("calories", calories, MAX_MEAL_CALORIES),
        ("protein", protein, MAX_MACRO_GRAMS),
        ("carbs", carbs, MAX_MACRO_GRAMS),
        ("fat", fat, MAX_MACRO_GRAMS),
    )
    for name, value, ceiling in checks:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{name} must be a number, got {value!r}."
        if number != number or number in (float("inf"), float("-inf")):
            return f"{name} must be a real number, got {value!r}."
        if number < 0:
            return f"{name} cannot be negative (got {number:g})."
        if number > ceiling:
            return (
                f"{number:g} {name} is not a plausible single meal (limit "
                f"{ceiling:g}). Check the portion with the user before logging — "
                "this usually means a quantity was misread."
            )
    return None


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
    """Log a NEW meal. Get macros from lookup_nutrition first.

    One eating occasion = one call (a photo plus its caption is ONE meal).
    Never use this to fix an existing meal - that is update_meal.

    Args:
        meal_name: Short label, e.g. "2 parathas and chai".
        calories: Total kcal for the whole meal.
        protein: Grams.
        carbs: Grams.
        fat: Grams.
        description: Portion detail, e.g. "~2/3 of the box".
        meal_type: breakfast, lunch, dinner or snack.
        source: "text", "vision" or "vision+text".
        meal_date: OMIT unless the user says the meal was on another day
            ("forgot to log yesterday's dinner"). Never copy a date from
            get_meals - "same as yesterday" is eaten TODAY.
    """
    meal_type = meal_type.lower().strip()
    if meal_type not in MEAL_TYPES:
        meal_type = "snack"

    error = _validate_macros(calories, protein, carbs, fat)
    if error:
        return {"error": error}

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
        from src.db.schema import connection

        with connection() as conn:
            conn.execute(
                "UPDATE meals SET meal_date = ? WHERE id = ? AND user_id = ?",
                (meal_date, meal["id"], get_user_id()),
            )
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
    """Retrieve logged meals, newest first. Use name_contains to find the meal a
    correction refers to; period="yesterday" for "same as yesterday".

    Args:
        period: "today", "yesterday", "week" or "all".
        start_date: Local YYYY-MM-DD; overrides period.
        end_date: Local YYYY-MM-DD; overrides period.
        name_contains: Text in name or description, e.g. "roti".
        limit: Max rows.
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


@tool("update_meal")
def update_meal(
    meal_id: int,
    meal_name: Optional[str] = None,
    calories: Optional[float] = None,
    protein: Optional[float] = None,
    carbs: Optional[float] = None,
    fat: Optional[float] = None,
    description: Optional[str] = None,
    meal_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Correct a meal that is ALREADY logged; edits the row in place.

    For "actually that was 3 rotis not 2": find it with get_meals, re-price the
    whole corrected meal with lookup_nutrition, then pass NEW TOTAL values here.
    Never log_meal a fix - that double-counts the day.

    Args:
        meal_id: From get_meals.
        meal_name: Corrected label, e.g. "3 rotis".
        calories: New TOTAL kcal (replaces, does not add).
        protein: New total grams.
        carbs: New total grams.
        fat: New total grams.
        description: Corrected portion detail.
        meal_type: breakfast, lunch, dinner or snack.
    """
    if meal_type:
        meal_type = meal_type.lower().strip()
        if meal_type not in MEAL_TYPES:
            meal_type = None

    error = _validate_macros(calories, protein, carbs, fat)
    if error:
        return {"error": error}

    updated = update_meal_row(
        user_id=get_user_id(),
        meal_id=int(meal_id),
        meal_name=meal_name,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        description=description,
        meal_type=meal_type,
    )
    if updated is None:
        return {
            "error": f"No meal with id {meal_id}. Call get_meals to find the right one.",
        }
    return {
        "updated": _summarize(updated),
        "daily_totals": daily_totals(get_user_id(), updated["meal_date"]),
    }


@tool("delete_meal")
def delete_meal(meal_id: int) -> Dict[str, Any]:
    """Remove a logged meal entirely ("delete that"). For amount fixes use update_meal.

    Args:
        meal_id: From get_meals.
    """
    user_id = get_user_id()
    meal = get_meal(user_id, int(meal_id))
    if meal is None:
        return {"error": f"No meal with id {meal_id}. Call get_meals to find the right one."}

    delete_meal_row(user_id, int(meal_id))
    return {
        "deleted": _summarize(meal),
        "daily_totals": daily_totals(user_id, meal["meal_date"]),
    }
