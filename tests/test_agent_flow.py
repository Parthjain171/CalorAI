"""Graph-level behaviour under the scripted double (no network)."""

from src.agent import CalorAIAgent
from src.db.queries import daily_totals, list_meals, list_memories


def _agent() -> CalorAIAgent:
    return CalorAIAgent(persistent=False)


def test_one_message_one_row():
    a = _agent()
    a.chat("u", "had 2 parathas and chai for breakfast")
    rows = list_meals("u")
    assert len(rows) == 1 and rows[0]["meal_type"] == "breakfast"


def test_correction_replaces_instead_of_adding():
    a = _agent()
    a.chat("u", "had 2 rotis and dal for lunch")
    before = daily_totals("u")["calories"]
    a.chat("u", "actually that was 3 rotis not 2")
    rows = list_meals("u")
    assert len(rows) == 1
    assert daily_totals("u")["calories"] == before + 110  # one more roti, dal kept


def test_vague_message_asks_and_logs_nothing():
    a = _agent()
    reply = a.chat("u", "skipped lunch but grazed all afternoon")
    assert "?" in reply and list_meals("u") == []


def test_photo_with_caption_is_one_half_meal():
    a = _agent()
    a.chat("full", "", image_path="assets/sample_plate.png")
    a.chat("half", "half of this was my brother's", image_path="assets/sample_plate.png")
    assert len(list_meals("half")) == 1
    assert daily_totals("half")["calories"] == daily_totals("full")["calories"] / 2


def test_preference_is_memory_not_meal():
    a = _agent()
    a.chat("u", "i'm vegetarian btw")
    assert list_meals("u") == []
    assert any("veg" in m["value"] for m in list_memories("u"))


def test_streaming_yields_the_same_reply():
    a = _agent()
    streamed = "".join(a.stream_chat("u", "how am I doing?"))
    assert "cal" in streamed
