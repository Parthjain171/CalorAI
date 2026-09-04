"""Memory: tiers, relevance ranking, the cap, and prompt rendering."""

from src.memory.manager import format_for_prompt, recall, remember
from src.utils.config import reload_settings


def test_always_on_tiers_are_injected_regardless_of_query():
    remember("u", "diet", "vegetarian", "preference")
    remember("u", "protein_goal", "140g", "goal")
    remember("u", "usual_breakfast", "2 idli", "usual_meal")
    keys = {m["key"] for m in recall("u", "how am I doing")}
    assert keys == {"diet", "protein_goal", "usual_breakfast"}


def test_keyword_tiers_need_a_match():
    remember("u", "gym_days", "gyms monday and thursday", "habit")
    assert recall("u", "how am I doing") == []
    assert [m["key"] for m in recall("u", "gym day today")] == ["gym_days"]


def test_cap_is_enforced_always_on_first(monkeypatch):
    monkeypatch.setenv("CALORAI_MAX_MEMORIES", "3")
    reload_settings()
    try:
        remember("u", "diet", "vegetarian", "preference")
        remember("u", "goal", "140g", "goal")
        for i in range(5):
            remember("u", f"habit_{i}", f"habit number {i} about biscuits", "habit")
        out = recall("u", "biscuits")
        assert len(out) == 3
        assert {out[0]["key"], out[1]["key"]} == {"diet", "goal"}
    finally:
        monkeypatch.delenv("CALORAI_MAX_MEMORIES")
        reload_settings()


def test_unknown_category_is_coerced_and_key_normalised():
    saved = remember("u", "My Fav Snack!", "khakhra", "nonsense")
    assert saved["category"] == "fact"
    assert saved["key"] == "my_fav_snack"


def test_use_count_increments_on_recall():
    remember("u", "diet", "vegetarian", "preference")
    recall("u", "x")
    recall("u", "y")
    assert recall("u", "z")[0]["use_count"] >= 2


def test_prompt_block_rendering():
    assert format_for_prompt([]) == ""
    remember("u", "diet", "vegetarian", "preference")
    block = format_for_prompt(recall("u", ""))
    assert "WHAT YOU KNOW ABOUT THIS USER" in block
    assert "(preference) diet: vegetarian" in block
