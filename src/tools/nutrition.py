"""``lookup_nutrition`` — the only tool that knows what food is worth.

Four tiers, cheapest first:

1. **Seed table** (``nutrition_data.SEED_NUTRITION``) — in-process dict, ~0 ms.
2. **SQLite cache** — anything a lower tier has resolved before, on any past run.
3. **USDA FoodData Central** (``nutrition_api``) — measured data, one HTTP call
   per unknown food, fanned out concurrently, then cached.
4. **LLM estimator** — one small batched call for whatever no database lists.

The tool takes a *list* of foods rather than one. A message like "2 parathas and
chai" needs two lookups; batching them into a single tool call saves a whole
model round-trip versus letting the agent issue them one at a time, and lets the
estimator resolve every cache miss in one request instead of N.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.db.queries import get_cached_nutrition, put_cached_nutrition
from src.tools.nutrition_data import (
    ALIASES,
    DEFAULT_FALLBACK,
    FALLBACK_BY_KEYWORD,
    SEED_NUTRITION,
)
from src.utils.config import settings

_QUANTITY_PREFIX = re.compile(
    r"^\s*(\d+(\.\d+)?|a|an|one|two|three|four|five|some|half|few)\s+", re.I
)
_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalize_food(name: str) -> str:
    """Reduce a free-text food mention to a canonical seed/cache key."""
    text = _PUNCT.sub(" ", name.lower()).strip()
    while _QUANTITY_PREFIX.match(text):
        text = _QUANTITY_PREFIX.sub("", text, count=1).strip()
    text = re.sub(r"\s+", " ", text)

    for candidate in (text, ALIASES.get(text, "")):
        if candidate in SEED_NUTRITION:
            return candidate
    if text in ALIASES:
        return ALIASES[text]

    singular = text[:-1] if text.endswith("s") and not text.endswith("ss") else text
    if singular in SEED_NUTRITION:
        return singular
    if singular in ALIASES:
        return ALIASES[singular]

    # Longest seed key that appears inside the phrase: "plate of chicken biryani"
    # should beat the shorter "biryani" match.
    matches = [key for key in SEED_NUTRITION if key in text or key in singular]
    if matches:
        return max(matches, key=len)
    return singular or text


def _from_seed(key: str) -> Optional[Tuple[str, float, float, float, float]]:
    return SEED_NUTRITION.get(key)


def _heuristic(key: str) -> Tuple[float, float, float, float]:
    """Last-resort estimate when no model is reachable."""
    for keyword, macros in FALLBACK_BY_KEYWORD.items():
        if keyword in key:
            return macros
    return DEFAULT_FALLBACK


def _estimate_with_llm(foods: List[str]) -> Dict[str, Dict[str, float]]:
    """Ask a small model for per-serving macros for foods we have never seen.

    One request for all misses. Returns ``{}`` on any failure so the caller can
    fall back to the heuristic — a wrong-ish number that gets logged beats an
    error message in a texting UX.
    """
    if not foods:
        return {}
    try:
        from src.models.text_model import get_chat_model
        from src.utils.latency import measure

        model = get_chat_model(settings.nutrition_model, temperature=0, max_tokens=600)
        prompt = (
            "You are a nutrition database. For each food below give macros for ONE "
            "typical single serving as eaten in India.\n"
            "Reply with ONLY a JSON object, no prose, shaped like:\n"
            '{"<food>": {"serving": "1 bowl", "calories": 0, "protein": 0, '
            '"carbs": 0, "fat": 0}}\n\n'
            "Foods: " + ", ".join(foods)
        )
        with measure("nutrition_llm", misses=len(foods)):
            raw = model.invoke(prompt).content
        if isinstance(raw, list):  # Anthropic returns a content-block list
            raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
        match = re.search(r"\{.*\}", str(raw), re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:  # noqa: BLE001 - degrade to heuristic, never break the turn
        return {}


def resolve_nutrition(food: str, servings: float = 1.0) -> Dict[str, Any]:
    """Resolve one food to macros, scaled by ``servings``. Used by tools and tests."""
    return _resolve_many([(food, servings)])[0]


def _resolve_many(items: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    resolved: List[Optional[Dict[str, Any]]] = [None] * len(items)
    misses: Dict[str, List[int]] = {}

    for index, (food, _servings) in enumerate(items):
        key = normalize_food(food)
        seed = _from_seed(key)
        if seed:
            serving, cal, pro, carb, fat = seed
            resolved[index] = {
                "key": key, "serving": serving, "calories": cal,
                "protein": pro, "carbs": carb, "fat": fat, "source": "seed",
            }
            continue
        cached = get_cached_nutrition(key)
        if cached:
            resolved[index] = {
                "key": key, "serving": cached["quantity"], "calories": cached["calories"],
                "protein": cached["protein"], "carbs": cached["carbs"],
                "fat": cached["fat"], "source": "cache",
            }
            continue
        misses.setdefault(key, []).append(index)

    if misses:
        # Tier 3: a real nutrition database, fanned out concurrently. Measured
        # numbers beat a model's guess, and each hit is cached for good.
        from src.tools.nutrition_api import lookup_usda_many

        for key, hit in lookup_usda_many(list(misses)).items():
            put_cached_nutrition(
                key, key, hit["serving"], hit["calories"], hit["protein"],
                hit["carbs"], hit["fat"], "usda",
            )
            for index in misses[key]:
                resolved[index] = {
                    "key": key, "serving": hit["serving"], "calories": hit["calories"],
                    "protein": hit["protein"], "carbs": hit["carbs"],
                    "fat": hit["fat"], "source": "usda",
                }
        misses = {k: v for k, v in misses.items() if resolved[v[0]] is None}

    if misses:
        # Tier 4: the LLM, only for what no database lists.
        estimates = _estimate_with_llm(list(misses))
        lowered = {str(k).lower(): v for k, v in estimates.items()}
        for key, indices in misses.items():
            guess = lowered.get(key) or lowered.get(key.rstrip("s")) or {}
            if guess:
                serving = str(guess.get("serving", "1 serving"))
                cal = float(guess.get("calories", 0) or 0)
                pro = float(guess.get("protein", 0) or 0)
                carb = float(guess.get("carbs", 0) or 0)
                fat = float(guess.get("fat", 0) or 0)
                source = "llm"
            else:
                cal, pro, carb, fat = _heuristic(key)
                serving, source = "1 serving (estimated)", "fallback"
            # Cache LLM answers only; heuristics are too coarse to freeze in.
            if source == "llm":
                put_cached_nutrition(key, key, serving, cal, pro, carb, fat, source)
            for index in indices:
                resolved[index] = {
                    "key": key, "serving": serving, "calories": cal, "protein": pro,
                    "carbs": carb, "fat": fat, "source": source,
                }

    output: List[Dict[str, Any]] = []
    for (food, servings), base in zip(items, resolved):
        assert base is not None
        factor = float(servings)
        output.append({
            "food": food,
            "matched_as": base["key"],
            "servings": round(factor, 2),
            "serving_size": base["serving"],
            "calories": round(base["calories"] * factor, 1),
            "protein": round(base["protein"] * factor, 1),
            "carbs": round(base["carbs"] * factor, 1),
            "fat": round(base["fat"] * factor, 1),
            "source": base["source"],
        })
    return output


class FoodQuery(BaseModel):
    """One food to price out."""

    food: str = Field(description="Food name, e.g. 'paratha', 'chicken biryani'.")
    servings: float = Field(
        default=1.0,
        description=(
            "How many standard servings. Use fractions for approximate portions: "
            "'two thirds of the box' -> 0.67, 'half of this' -> 0.5, '2 rotis' -> 2."
        ),
    )


@tool("lookup_nutrition")
def lookup_nutrition(items: List[FoodQuery]) -> Dict[str, Any]:
    """Get calories and macros for one or more foods, scaled to the portion eaten.

    Always pass EVERY food from the message in a single call — this is batched
    and one call is much faster than several. Returns per-item macros plus a
    combined `total` you can pass straight to log_meal.
    """
    pairs = [(item.food, item.servings) for item in items]
    results = _resolve_many(pairs)
    total = {
        macro: round(sum(r[macro] for r in results), 1)
        for macro in ("calories", "protein", "carbs", "fat")
    }
    return {"items": results, "total": total}
