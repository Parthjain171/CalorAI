"""Tool registry. The agent binds exactly what this module exports."""

from src.tools.meal_tools import delete_meal, get_meals, log_meal, update_meal
from src.tools.memory import recall_memory, store_memory
from src.tools.nutrition import lookup_nutrition
from src.tools.totals import get_daily_totals

ALL_TOOLS = [
    lookup_nutrition,
    log_meal,
    get_meals,
    update_meal,
    delete_meal,
    get_daily_totals,
    store_memory,
    recall_memory,
]

__all__ = [
    "ALL_TOOLS",
    "lookup_nutrition",
    "log_meal",
    "get_meals",
    "update_meal",
    "delete_meal",
    "get_daily_totals",
    "store_memory",
    "recall_memory",
]
