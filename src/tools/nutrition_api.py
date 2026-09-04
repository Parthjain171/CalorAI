"""USDA FoodData Central client: real nutrition data for foods the seed table
does not know.

Sits between the SQLite cache and the LLM estimator in ``lookup_nutrition``.
A database hit is better than a model guess: it is measured, reproducible, and
free. The LLM remains the tier of last resort for dishes no database lists.

Uses only the standard library (``urllib``) so it adds no dependency, and every
failure mode (timeout, rate limit, no match) returns ``None`` so the caller
falls through to the next tier instead of breaking the turn.

Key: ``USDA_API_KEY``. The tier is **off unless a key is set**. A free personal
key takes a minute at https://fdc.nal.usda.gov/api-key-signup.html and allows
1,000 requests/hour. ``DEMO_KEY`` also works but is capped at **10 requests per
hour** (measured: ``X-Ratelimit-Limit: 10``), which is enough to see the tier
work and nothing more - so it is not the default. Results are cached in SQLite,
so each distinct food costs one request ever.

Two guards keep a limited or offline key from hurting latency:

* a **circuit breaker** - after a 429 the tier stands down for a cooldown
  instead of paying a round-trip on every miss just to be refused again;
* an in-process **negative cache** - a food the database does not list is not
  asked about again this session.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.utils import config  # module attribute so reload_settings() is honoured

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

RATE_LIMIT_COOLDOWN_SECONDS = 15 * 60

_state_lock = threading.Lock()
_rate_limited_until = 0.0
_no_match: set[str] = set()


def _available() -> bool:
    """True if the tier has a key and is not sitting out a rate-limit cooldown."""
    if not config.settings.usda_api_key:
        return False
    with _state_lock:
        return time.monotonic() >= _rate_limited_until


def _note_rate_limited() -> None:
    """Open the circuit: skip the network for the cooldown period."""
    global _rate_limited_until
    with _state_lock:
        _rate_limited_until = time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS


def reset_circuit() -> None:
    """Clear breaker and negative cache (tests)."""
    global _rate_limited_until
    with _state_lock:
        _rate_limited_until = 0.0
        _no_match.clear()

# Generic (non-branded) datasets only. FNDDS describes food as eaten
# ("Bread, paratha"), which matches what people text far better than
# "Brand X frozen paratha 400g".
_DATA_TYPES = "Survey (FNDDS),Foundation,SR Legacy"
_DATATYPE_RANK = {"Survey (FNDDS)": 0, "Foundation": 1, "SR Legacy": 2}

_NUTRIENT_FIELDS = {
    "Energy": "calories",
    "Protein": "protein",
    "Carbohydrate, by difference": "carbs",
    "Total lipid (fat)": "fat",
}

_STOPWORDS = frozenset({"the", "and", "with", "raw", "cooked", "plain", "some"})
_TOKEN = re.compile(r"[a-z]+")

# Search results report per 100 g. Without a portion lookup (a second request
# per food) 100 g is the honest serving to report; the tool result carries the
# serving size so the model can scale it.
SERVING_GRAMS = 100.0


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _nutrients(food: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Pull kcal + macros per 100 g from one search hit.

    Energy can appear twice (kcal and kJ); only the KCAL row is taken. A hit
    missing calories is useless and is rejected.
    """
    out: Dict[str, float] = {}
    for entry in food.get("foodNutrients", []):
        name = entry.get("nutrientName")
        field = _NUTRIENT_FIELDS.get(name)
        if field is None:
            continue
        if field == "calories" and str(entry.get("unitName", "")).upper() != "KCAL":
            continue
        try:
            out[field] = float(entry.get("value", 0) or 0)
        except (TypeError, ValueError):
            continue
    if "calories" not in out or out["calories"] <= 0:
        return None
    for field in ("protein", "carbs", "fat"):
        out.setdefault(field, 0.0)
    return out


def parse_search_response(payload: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Choose the best hit for ``query`` from a search response, or ``None``.

    Guard against the search engine's enthusiasm: every meaningful token of the
    query must appear in the hit's description, otherwise "chai" could resolve
    to "Chard, cooked". Ties are broken by dataset quality (FNDDS first).
    """
    wanted = _tokens(query)
    if not wanted:
        return None

    candidates = []
    for food in payload.get("foods", []) or []:
        description = str(food.get("description", "")).lower()
        if not all(token in description for token in wanted):
            continue
        macros = _nutrients(food)
        if macros is None:
            continue
        rank = _DATATYPE_RANK.get(str(food.get("dataType")), 9)
        candidates.append((rank, len(description), food, macros))

    if not candidates:
        return None
    rank, _, food, macros = min(candidates, key=lambda c: (c[0], c[1]))
    scale = SERVING_GRAMS / 100.0
    return {
        "fdc_id": food.get("fdcId"),
        "description": food.get("description"),
        "data_type": food.get("dataType"),
        "serving": f"{SERVING_GRAMS:g} g",
        "calories": round(macros["calories"] * scale, 1),
        "protein": round(macros["protein"] * scale, 1),
        "carbs": round(macros["carbs"] * scale, 1),
        "fat": round(macros["fat"] * scale, 1),
    }


def lookup_usda(food: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Search FoodData Central for one food. ``None`` on any failure or no match."""
    if not _available():
        return None
    with _state_lock:
        if food in _no_match:
            return None

    params = urllib.parse.urlencode({
        "api_key": config.settings.usda_api_key,
        "query": food,
        "pageSize": 8,
        "dataType": _DATA_TYPES,
    })
    request = urllib.request.Request(
        f"{USDA_SEARCH_URL}?{params}", headers={"User-Agent": "calorai-agent/0.1"}
    )
    try:
        from src.utils.latency import measure

        with measure("nutrition_api", food=food):
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _note_rate_limited()
        return None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None  # offline, malformed - fall through to the LLM

    hit = parse_search_response(payload, food)
    if hit is None:
        with _state_lock:
            _no_match.add(food)
    return hit


def lookup_usda_many(foods: List[str], timeout: float = 5.0) -> Dict[str, Dict[str, Any]]:
    """Look up several foods concurrently. Returns only the ones that matched.

    Cache misses are rare after the first few days of use, but when a message
    has three unknown dishes, three sequential HTTP calls would cost ~1.5 s.
    Fanning them out keeps the tier at the cost of one.
    """
    if not foods or not _available():
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(foods))) as pool:
        for food, hit in zip(foods, pool.map(lambda f: lookup_usda(f, timeout), foods)):
            if hit:
                results[food] = hit
    return results
