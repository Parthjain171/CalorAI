"""Memory: what gets kept, when it is written, and how it is retrieved.

Not conversation history. This is a small set of durable facts about a person
that should still be true next month - and it is deliberately small, because
memory that grows without bound stops being memory and becomes context bloat.

**Categories** (a memory's category decides how it is retrieved):

===============  =========================================  ==================
category         examples                                   retrieval
===============  =========================================  ==================
``preference``   vegetarian, allergic to peanuts, no beef    always injected
``goal``         140g protein/day, cutting to 1800 cal       always injected
``usual_meal``   usual breakfast = 2 idli + sambar           always injected
``habit``        skips lunch on weekdays, gyms at 6am        keyword-matched
``fact``         anything else worth keeping                 keyword-matched
===============  =========================================  ==================

The first three tiers are *always* injected because they are few (a handful per
user), they change the answer even when the message does not mention them, and
getting them wrong is the visible failure - suggesting chicken to a vegetarian
is worse than any amount of latency. The keyword-matched tiers are where volume
accumulates, so those must earn their place in the prompt.

**Bloat control**, in three layers:

1. Write-side selectivity - only the five categories above are storable, and
   the ``store_memory`` tool description tells the model what *not* to keep.
   Individual meals are never memories; they are rows in ``meals``.
2. Keyed upsert - ``UNIQUE(user_id, key)`` means restating a preference
   overwrites it. Saying "I'm vegetarian" ten times yields one row.
3. A hard cap - at most ``CALORAI_MAX_MEMORIES`` (default 8) reach the prompt on
   any turn, ranked by tier, then keyword score, then how recently used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from src.db.queries import list_memories, touch_memories, upsert_memory
# Module attribute, not `from ... import settings`: reload_settings() rebinds
# config.settings, and a name imported at module load would keep the old copy.
from src.utils import config

CATEGORIES = ("preference", "goal", "usual_meal", "habit", "fact")
ALWAYS_INJECTED = ("preference", "goal", "usual_meal")

# Words too common to signal relevance when matching a message to a memory.
_STOPWORDS = frozenset(
    """a an and are as at be but by for had has have i i'm im in is it its me my
    of on or so that the this to was were what with you your had have some just
    like really very today yesterday now then did do does""".split()
)
_TOKEN = re.compile(r"[a-z']+")

MAX_VALUE_CHARS = 200


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def remember(user_id: str, key: str, value: str, category: str = "fact") -> Dict[str, Any]:
    """Write a durable fact, coercing anything unexpected into a safe shape."""
    category = (category or "fact").strip().lower()
    if category not in CATEGORIES:
        category = "fact"
    key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_") or "fact"
    value = value.strip()[:MAX_VALUE_CHARS]
    return upsert_memory(user_id, key, value, category)


def _score(memory: Dict[str, Any], query_tokens: set[str]) -> float:
    """Relevance of one memory to the current message."""
    text_tokens = _tokens(f"{memory['key'].replace('_', ' ')} {memory['value']}")
    overlap = len(text_tokens & query_tokens)
    if not overlap:
        return 0.0
    # Recently and frequently used memories break ties - a "usual breakfast" the
    # user leans on daily should outrank a fact mentioned once.
    return overlap + min(memory.get("use_count", 0), 5) * 0.1


def recall(
    user_id: str, query: str = "", limit: int | None = None
) -> List[Dict[str, Any]]:
    """Return the memories worth putting in front of the model this turn."""
    cap = limit or config.settings.max_memories
    everything = list_memories(user_id)
    if not everything:
        return []

    query_tokens = _tokens(query)
    always = [m for m in everything if m["category"] in ALWAYS_INJECTED]
    others = [m for m in everything if m["category"] not in ALWAYS_INJECTED]

    scored = sorted(
        ((m, _score(m, query_tokens)) for m in others),
        key=lambda pair: pair[1],
        reverse=True,
    )
    matched = [m for m, score in scored if score > 0]

    selected = (always + matched)[:cap]
    touch_memories(user_id, [m["key"] for m in selected])
    return selected


def format_for_prompt(memories: Sequence[Dict[str, Any]]) -> str:
    """Render memories as a compact prompt block. Empty string when there are none."""
    if not memories:
        return ""
    lines = [
        f"- ({m['category']}) {m['key'].replace('_', ' ')}: {m['value']}"
        for m in memories
    ]
    return (
        "WHAT YOU KNOW ABOUT THIS USER\n"
        + "\n".join(lines)
        + "\nUse these without being asked. Never suggest food that conflicts with "
        "a stated dietary preference."
    )
