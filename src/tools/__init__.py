"""Tool registry. The agent binds exactly what this module exports."""

from src.tools.meal_tools import get_meals, log_meal
from src.tools.nutrition import lookup_nutrition
from src.tools.totals import get_daily_totals

ALL_TOOLS = [
    lookup_nutrition,
    log_meal,
    get_meals,
    get_daily_totals,
]

__all__ = [
    "ALL_TOOLS",
    "lookup_nutrition",
    "log_meal",
    "get_meals",
    "get_daily_totals",
]
