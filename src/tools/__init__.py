"""Tool registry. The agent binds exactly what this module exports."""

from src.tools.meal_tools import get_meals, log_meal
from src.tools.nutrition import lookup_nutrition

ALL_TOOLS = [
    lookup_nutrition,
    log_meal,
    get_meals,
]

__all__ = ["ALL_TOOLS", "lookup_nutrition", "log_meal", "get_meals"]
