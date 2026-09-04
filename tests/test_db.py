"""Data layer: inserts, in-place updates, deletes, totals, isolation."""

from src.db import queries as q
from src.utils.config import local_date_str


def test_insert_then_totals():
    q.insert_meal("u", "2 rotis", 220, 6, 40, 5, meal_type="lunch")
    totals = q.daily_totals("u")
    assert totals["meal_count"] == 1
    assert totals["calories"] == 220
    assert totals["protein"] == 6


def test_update_is_in_place_not_duplicate():
    meal = q.insert_meal("u", "2 rotis", 220, 6, 40, 5)
    q.update_meal_row("u", meal["id"], meal_name="3 rotis", calories=330, protein=9)
    rows = q.list_meals("u")
    assert len(rows) == 1
    assert rows[0]["id"] == meal["id"]
    assert rows[0]["meal_name"] == "3 rotis"
    assert q.daily_totals("u")["calories"] == 330


def test_update_ignores_unknown_and_none_fields():
    meal = q.insert_meal("u", "dal", 150, 9, 20, 4)
    q.update_meal_row("u", meal["id"], calories=None, bogus="x")
    assert q.get_meal("u", meal["id"])["calories"] == 150


def test_update_missing_meal_returns_none():
    assert q.update_meal_row("u", 9999, calories=1) is None


def test_delete_adjusts_totals():
    a = q.insert_meal("u", "a", 100)
    q.insert_meal("u", "b", 200)
    assert q.delete_meal_row("u", a["id"]) is True
    assert q.delete_meal_row("u", a["id"]) is False
    assert q.daily_totals("u")["calories"] == 200


def test_users_are_isolated():
    q.insert_meal("alice", "x", 100)
    q.insert_meal("bob", "y", 500)
    assert q.daily_totals("alice")["calories"] == 100
    assert q.daily_totals("bob")["calories"] == 500
    assert q.get_meal("alice", q.list_meals("bob")[0]["id"]) is None


def test_name_contains_finds_the_right_row():
    q.insert_meal("u", "2 parathas and chai", 610)
    q.insert_meal("u", "2 rotis and dal", 370)
    hits = q.list_meals("u", name_contains="roti")
    assert [h["meal_name"] for h in hits] == ["2 rotis and dal"]


def test_date_window_filters():
    q.insert_meal("u", "today", 100)
    y = local_date_str(-1)
    assert q.list_meals("u", start_date=y, end_date=y) == []
    assert len(q.list_meals("u", start_date=local_date_str(), end_date=local_date_str())) == 1


def test_memory_upsert_dedupes():
    q.upsert_memory("u", "diet", "vegetarian", "preference")
    q.upsert_memory("u", "diet", "vegan", "preference")
    mems = q.list_memories("u")
    assert len(mems) == 1
    assert mems[0]["value"] == "vegan"


def test_nutrition_cache_roundtrip():
    q.put_cached_nutrition("thepla@1", "thepla", "1 piece", 150, 4, 22, 5, "llm")
    hit = q.get_cached_nutrition("thepla@1")
    assert hit["calories"] == 150 and hit["source"] == "llm"
    assert q.get_cached_nutrition("nope") is None
