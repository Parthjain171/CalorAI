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


def test_bounded_history_never_trims_the_current_turn(monkeypatch):
    """A turn larger than the whole budget must still reach the model intact.

    Regression for the reply "I'm ready to log your meals. What did you eat?"
    arriving right after a meal was logged: the trimmer dropped the current
    turn - human message, tool calls and results - and the model answered from
    the system prompt alone.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    import dataclasses

    from src import agent as agent_module

    # Settings is frozen, so swap the module's reference for a modified copy.
    monkeypatch.setattr(
        agent_module, "settings", dataclasses.replace(agent_module.settings, max_history_tokens=50)
    )

    big = "x " * 400
    earlier = [HumanMessage("had 2 rotis"), AIMessage("Logged 2 rotis, 220 cal.")]
    current = [
        HumanMessage("half of this was my brother's\n\n[VISION] " + big),
        AIMessage("", tool_calls=[{"name": "log_meal", "args": {}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(big, tool_call_id="c1"),
    ]
    seen = agent_module._bounded_history(earlier + current)
    assert seen[-len(current):] == current
    assert earlier[0] not in seen  # over budget, so older turns are what gets cut


def test_vision_note_spells_out_the_caption_portion():
    from src.models.vision_model import format_vision_note

    analysis = {
        "foods": [{"food": "rice", "servings": 1.0}], "confidence": 0.9,
        "description": "", "question": None, "model": "m", "error": None,
    }
    assert "multiply EVERY serving above by 0.5" in format_vision_note(analysis, "half of this was my brother's")
    assert "by 0.67" in format_vision_note(analysis, "ate two thirds of it")
    assert "multiply" not in format_vision_note(analysis, "lunch today")
