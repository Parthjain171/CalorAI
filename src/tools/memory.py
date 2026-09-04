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
    """Save a DURABLE fact about the user — something still true next month.

    Store: dietary preferences ("I'm vegetarian", "allergic to peanuts"),
    nutrition goals ("targeting 140g protein"), what a recurring meal means for
    them ("my usual breakfast" = 2 idli and sambar), and standing habits.

    Do NOT store: individual meals (those are log_meal), one-off comments, how
    they felt, or anything you would not want repeated back in three weeks.

    Reuse an existing key to update a fact rather than inventing a near-duplicate
    — "diet" restated overwrites the old value instead of stacking up.

    Args:
        key: Short stable snake_case identifier, e.g. "diet", "protein_goal",
            "usual_breakfast", "avoids".
        value: The fact in plain words, e.g. "vegetarian, eats eggs".
        category: One of preference, goal, usual_meal, habit, fact.
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
    """Look up what you know about this user.

    Use it when the message leans on a stored fact you need the exact value of —
    "my usual", "the same as always", "am I hitting my protein goal". Dietary
    preferences and goals are already in your context; you do not need this to
    check those.

    Args:
        query: What you are looking for, e.g. "usual breakfast", "diet".
        limit: Max memories to return.
    """
    memories = recall(get_user_id(), query, limit=limit)
    return {
        "count": len(memories),
        "memories": [
            {"key": m["key"], "value": m["value"], "category": m["category"]}
            for m in memories
        ],
    }
