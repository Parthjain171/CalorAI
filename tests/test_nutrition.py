"""Nutrition resolution: normalisation, scaling, tiers, USDA parsing, validation."""

from src.db.queries import get_cached_nutrition, put_cached_nutrition
from src.tools.meal_tools import _validate_macros, log_meal, update_meal
from src.tools.nutrition import lookup_nutrition, normalize_food, resolve_nutrition
from src.tools import nutrition_api
from src.tools.nutrition_api import parse_search_response
from src.utils.config import reload_settings
from src.utils.user_context import user_scope


def test_normalize_strips_quantities_and_plurals():
    assert normalize_food("2 parathas") == "paratha"
    assert normalize_food("three rotis") == "roti"
    assert normalize_food("Chapatis!") == "roti"          # alias + plural
    assert normalize_food("dahi") == "curd"                 # alias


def test_normalize_prefers_longest_seed_match():
    assert normalize_food("a plate of chicken biryani") == "chicken biryani"
    assert normalize_food("leftover biryani") == "biryani"


def test_servings_scale_linearly():
    one = resolve_nutrition("roti", 1)
    three = resolve_nutrition("roti", 3)
    assert three["calories"] == one["calories"] * 3
    assert one["source"] == "seed"


def test_fractional_portion():
    assert resolve_nutrition("biryani", 0.5)["calories"] == resolve_nutrition("biryani", 1)["calories"] / 2


def test_batched_tool_returns_total():
    out = lookup_nutrition.invoke({"items": [
        {"food": "paratha", "servings": 2}, {"food": "chai", "servings": 1},
    ]})
    assert len(out["items"]) == 2
    assert out["total"]["calories"] == sum(i["calories"] for i in out["items"])


def test_cache_tier_is_used_before_estimation():
    put_cached_nutrition("thepla", "thepla", "1 piece", 150, 4, 22, 5, "llm")
    hit = resolve_nutrition("thepla", 2)
    assert hit["source"] == "cache"
    assert hit["calories"] == 300


def test_unknown_food_falls_back_without_crashing():
    # USDA disabled in tests and the mock "LLM" returns no JSON -> heuristic.
    hit = resolve_nutrition("xyzzy mystery dish", 1)
    assert hit["calories"] > 0
    assert hit["source"] == "fallback"
    assert get_cached_nutrition("xyzzy mystery dish") is None  # heuristics are not frozen in


USDA_FIXTURE = {
    "foods": [
        {   # kJ energy row first - must be skipped, not mistaken for kcal
            "fdcId": 1, "dataType": "SR Legacy", "description": "Bread, paratha, frozen",
            "foodNutrients": [
                {"nutrientName": "Energy", "unitName": "kJ", "value": 1360},
                {"nutrientName": "Energy", "unitName": "KCAL", "value": 326},
                {"nutrientName": "Protein", "unitName": "G", "value": 6.4},
                {"nutrientName": "Carbohydrate, by difference", "unitName": "G", "value": 45.4},
                {"nutrientName": "Total lipid (fat)", "unitName": "G", "value": 13.2},
            ],
        },
        {
            "fdcId": 2, "dataType": "Survey (FNDDS)", "description": "Bread, paratha",
            "foodNutrients": [
                {"nutrientName": "Energy", "unitName": "KCAL", "value": 326},
                {"nutrientName": "Protein", "unitName": "G", "value": 6.36},
                {"nutrientName": "Carbohydrate, by difference", "unitName": "G", "value": 45.35},
                {"nutrientName": "Total lipid (fat)", "unitName": "G", "value": 13.2},
            ],
        },
        {
            "fdcId": 3, "dataType": "Survey (FNDDS)", "description": "Chard, cooked",
            "foodNutrients": [{"nutrientName": "Energy", "unitName": "KCAL", "value": 20}],
        },
    ]
}


def test_usda_prefers_fndds_and_reads_kcal():
    hit = parse_search_response(USDA_FIXTURE, "paratha")
    assert hit["fdc_id"] == 2
    assert hit["calories"] == 326
    assert hit["protein"] == 6.4 or hit["protein"] == 6.36


def test_usda_rejects_hits_that_do_not_contain_the_query():
    assert parse_search_response(USDA_FIXTURE, "chai") is None


def test_usda_empty_payload():
    assert parse_search_response({}, "roti") is None
    assert parse_search_response({"foods": []}, "roti") is None


def test_usda_tier_is_off_without_a_key():
    assert nutrition_api._available() is False
    assert nutrition_api.lookup_usda_many(["paratha"]) == {}


def test_usda_circuit_breaker_and_negative_cache(monkeypatch):
    monkeypatch.setenv("USDA_API_KEY", "DEMO_KEY")
    reload_settings()
    nutrition_api.reset_circuit()
    calls = []

    class _Resp:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.body

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        if "limited" in request.full_url:
            raise nutrition_api.urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, None)
        return _Resp(b'{"foods": []}')

    monkeypatch.setattr(nutrition_api.urllib.request, "urlopen", fake_urlopen)
    try:
        assert nutrition_api._available() is True
        # No match -> negative-cached: second lookup makes no request.
        assert nutrition_api.lookup_usda("nomatch food") is None
        assert nutrition_api.lookup_usda("nomatch food") is None
        assert len(calls) == 1
        # 429 -> circuit opens: further lookups skip the network entirely.
        assert nutrition_api.lookup_usda("limited food") is None
        assert nutrition_api._available() is False
        nutrition_api.lookup_usda("another food")
        assert len(calls) == 2
    finally:
        nutrition_api.reset_circuit()
        monkeypatch.delenv("USDA_API_KEY")
        reload_settings()


def test_validation_bounds():
    assert _validate_macros(500, 20, 60, 15) is None
    assert "plausible" in _validate_macros(10_000_000, 0, 0, 0)
    assert "negative" in _validate_macros(-5, 0, 0, 0)
    assert _validate_macros(float("nan"), 0, 0, 0) is not None
    assert _validate_macros(None, None, None, None) is None


def test_log_meal_rejects_absurd_values_without_writing():
    with user_scope("u"):
        out = log_meal.invoke({"meal_name": "99999 rotis", "calories": 10_999_890})
        assert "error" in out
        ok = log_meal.invoke({"meal_name": "2 rotis", "calories": 220, "protein": 6})
        assert ok["daily_totals"]["meal_count"] == 1
        bad = update_meal.invoke({"meal_id": ok["logged"]["meal_id"], "calories": -1})
        assert "error" in bad
