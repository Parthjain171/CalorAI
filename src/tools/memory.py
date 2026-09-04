"""``store_memory`` and ``recall_memory``.

Writes are model-driven rather than automatic: the model decides a fact is
durable and calls ``store_memory``. Auto-summarising every turn into memory is
what produces the "remembers that you said hi on Tuesday" failure mode, so the
tool description is mostly a list of what *not* to keep.

Relevant memories are already injected into the system prompt each turn by the
agent's prepare step, so ``recall_memory`` exists for the case injection cannot
cover: an explicit lookup like "my usual", where the model needs the exact
stored value to act on rather than a hint.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from src.memory.manager import CATEGORIES, recall, remember
from src.utils.user_context import get_user_id


@tool("store_memory")
def store_memory(key: str, value: str, category: str = "fact") -> Dict[str, Any]:
    """Save a DURABLE fact about the user (still true next month): diet, allergies,
    goals ("140g protein"), what "my usual" means, standing habits.
    Not meals, moods or one-off remarks. Reuse a key to update it.

    Args:
        key: Stable snake_case id, e.g. "diet", "protein_goal", "usual_breakfast".
        value: The fact in plain words.
        category: preference, goal, usual_meal, habit or fact.
    """
    if category not in CATEGORIES:
        category = "fact"
    saved = remember(get_user_id(), key, value, category)
    return {
        "saved": {
            "key": saved["key"],
            "value": saved["value"],
            "category": saved["category"],
        }
    }


@tool("recall_memory")
def recall_memory(query: str = "", limit: int = 8) -> Dict[str, Any]:
    """Look up stored facts when you need the exact value - "my usual", "my
    protein goal". Diet and goals are already in your context.

    Args:
        query: e.g. "usual breakfast".
        limit: Max memories.
    """
    memories = recall(get_user_id(), query, limit=limit)
    return {
        "count": len(memories),
        "memories": [
            {"key": m["key"], "value": m["value"], "category": m["category"]}
            for m in memories
        ],
    }
