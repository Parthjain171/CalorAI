"""The eval set: the 11 conversations, with what "correct" means for each.

Every case asserts against the **database**, not against the wording of the
reply. A model that says "logged it!" and writes nothing must fail, and a model
that phrases things differently must still pass. Where a reply has to be checked
(a clarifying question, a quoted number) the assertion is on substance — did it
ask, does the number match the row — never on phrasing.

Numeric checks use ranges. The exact calorie figure depends on the model and the
seed table; what must hold is the *shape* of the outcome: one row not two, a
correction that replaces rather than adds, half a plate costing about half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.db.queries import daily_totals, insert_meal, list_meals, list_memories
from src.memory.manager import remember
from src.utils.config import local_date_str

SAMPLE_IMAGE = "assets/sample_plate.png"


@dataclass
class Turn:
    """One user message, optionally with a photo attached."""

    text: str = ""
    image: Optional[str] = None


@dataclass
class Case:
    """One scenario: optional setup, a script of turns, and a check."""

    id: str
    description: str
    turns: List[Turn]
    check: Callable[[str, List[str], Dict[str, Any]], List[str]]
    setup: Optional[Callable[[Any, str], Dict[str, Any]]] = None
    tags: List[str] = field(default_factory=list)


# --- helpers used by the checks ---------------------------------------------


def _today(user_id: str) -> List[Dict[str, Any]]:
    day = local_date_str()
    return list_meals(user_id, start_date=day, end_date=day)


def _expect_meals(user_id: str, count: int) -> List[str]:
    meals = _today(user_id)
    if len(meals) != count:
        found = ", ".join(f"{m['meal_name']!r}={m['calories']:g}cal" for m in meals)
        return [f"expected {count} meal(s) today, found {len(meals)} [{found}]"]
    return []


def _expect_calories(user_id: str, low: float, high: float) -> List[str]:
    total = daily_totals(user_id)["calories"]
    if not low <= total <= high:
        return [f"expected {low:g}-{high:g} cal today, got {total:g}"]
    return []


def _asked_a_question(replies: List[str]) -> List[str]:
    if "?" not in (replies[-1] if replies else ""):
        return [f"expected a clarifying question, got {replies[-1] if replies else ''!r}"]
    return []


def _mentions_number(reply: str, value: float, tolerance: float = 1.0) -> bool:
    """True if the reply quotes a number close to ``value``."""
    import re

    for found in re.findall(r"\d+(?:\.\d+)?", reply.replace(",", "")):
        if abs(float(found) - value) <= tolerance:
            return True
    return False


# --- the cases ---------------------------------------------------------------


def _check_logs_breakfast(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    problems = _expect_meals(user_id, 1) + _expect_calories(user_id, 300, 1000)
    meals = _today(user_id)
    if meals and meals[0]["meal_type"] != "breakfast":
        problems.append(f"expected meal_type breakfast, got {meals[0]['meal_type']!r}")
    if meals and meals[0]["protein"] <= 0:
        problems.append("expected protein to be recorded, got 0")
    return problems


def _check_approx_portion(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    # Two thirds of a biryani box: must log, and must not log a full portion.
    return _expect_meals(user_id, 1) + _expect_calories(user_id, 150, 600)


def _check_grazing(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    # Nothing is named, so nothing should be invented. Asking is the pass.
    return _expect_meals(user_id, 0) + _asked_a_question(replies)


def _setup_yesterday(agent: Any, user_id: str) -> Dict[str, Any]:
    yesterday = local_date_str(-1)
    for name, cal, pro, carb, fat, meal_type in [
        ("2 idli and sambar", 226, 9.0, 39.0, 3.6, "breakfast"),
        ("rajma chawal", 415, 16.2, 75.0, 4.4, "lunch"),
    ]:
        meal = insert_meal(user_id, name, cal, pro, carb, fat, meal_type=meal_type)
        from src.db.schema import get_connection

        conn = get_connection()
        conn.execute("UPDATE meals SET meal_date = ? WHERE id = ?", (yesterday, meal["id"]))
        conn.commit()
    return {"yesterday_calories": daily_totals(user_id, yesterday)["calories"]}


def _check_same_as_yesterday(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    meals = _today(user_id)
    if not meals:
        return ["expected yesterday's meals to be re-logged for today, found none"]
    expected = ctx.get("yesterday_calories", 0)
    total = daily_totals(user_id)["calories"]
    if abs(total - expected) > max(60.0, expected * 0.2):
        return [f"expected roughly {expected:g} cal (same as yesterday), got {total:g}"]
    return []


def _check_correction(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    """The differentiator: an update, never a second insert."""
    problems = _expect_meals(user_id, 1)
    total = daily_totals(user_id)["calories"]
    # 3 rotis ~330. Two rows would land near 220+330=550.
    if total > 480:
        problems.append(
            f"totals look double-counted: {total:g} cal — a correction must "
            "replace the meal, not add a second one"
        )
    meals = _today(user_id)
    if meals and not _mentions_number(meals[0]["meal_name"], 3, 0.0):
        problems.append(f"expected the row to describe 3 rotis, got {meals[0]['meal_name']!r}")
    return problems


def _setup_known_day(agent: Any, user_id: str) -> Dict[str, Any]:
    insert_meal(user_id, "2 parathas and chai", 610, 12.5, 76.0, 27.0, meal_type="breakfast")
    insert_meal(user_id, "rajma chawal", 415, 16.2, 75.0, 4.4, meal_type="lunch")
    return {"totals": daily_totals(user_id)}


def _check_protein_query(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    protein = daily_totals(user_id)["protein"]
    if not _mentions_number(replies[-1], protein, 1.0):
        return [f"expected the reply to quote {protein:g}g protein, got {replies[-1]!r}"]
    return _expect_meals(user_id, 2)


def _check_calorie_query(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    calories = daily_totals(user_id)["calories"]
    if not _mentions_number(replies[-1], calories, 2.0):
        return [f"expected the reply to quote {calories:g} cal, got {replies[-1]!r}"]
    return _expect_meals(user_id, 2)


def _check_photo(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    problems = _expect_meals(user_id, 1)
    meals = _today(user_id)
    if meals and "vision" not in meals[0]["source"]:
        problems.append(f"expected source to record vision, got {meals[0]['source']!r}")
    if meals and meals[0]["calories"] <= 0:
        problems.append("photo logged with no calories")
    return problems


def _setup_full_plate(agent: Any, user_id: str) -> Dict[str, Any]:
    """Log the same photo WITHOUT a caption, as a companion user, for comparison."""
    reference_user = f"{user_id}__reference"
    agent.chat(reference_user, "", image_path=SAMPLE_IMAGE)
    return {"full_plate_calories": daily_totals(reference_user)["calories"]}


def _check_photo_half(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    """Photo + "half of this was my brother's" -> ONE meal, about half the plate."""
    problems = _expect_meals(user_id, 1)
    full = ctx.get("full_plate_calories", 0)
    total = daily_totals(user_id)["calories"]
    if total <= 0:
        problems.append("nothing was logged for the photo")
    elif full and total > full * 0.75:
        problems.append(
            f"expected about half of the {full:g} cal plate, got {total:g} — "
            "the caption did not reduce the portion"
        )
    return problems


def _setup_usual(agent: Any, user_id: str) -> Dict[str, Any]:
    remember(user_id, "usual_breakfast", "2 idli and sambar", "usual_meal")
    return {}


def _check_usual(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    problems = _expect_meals(user_id, 1)
    meals = _today(user_id)
    if meals:
        text = f"{meals[0]['meal_name']} {meals[0]['description']}".lower()
        if "idli" not in text and "sambar" not in text:
            problems.append(
                f"expected the stored usual (idli/sambar) to be logged, got "
                f"{meals[0]['meal_name']!r}"
            )
    return problems


def _check_vegetarian(user_id: str, replies: List[str], ctx: Dict[str, Any]) -> List[str]:
    memories = list_memories(user_id)
    matched = [
        m for m in memories
        if "veg" in m["value"].lower() and m["category"] in ("preference", "fact")
    ]
    problems: List[str] = []
    if not matched:
        stored = ", ".join(f"{m['key']}={m['value']}" for m in memories) or "nothing"
        problems.append(f"expected a vegetarian preference in memory, stored: {stored}")
    problems += _expect_meals(user_id, 0)  # a statement about diet is not a meal
    return problems


CASES: List[Case] = [
    Case(
        id="01_log_simple_meal",
        description="'had 2 parathas and chai for breakfast' -> logs with macros",
        turns=[Turn("had 2 parathas and chai for breakfast")],
        check=_check_logs_breakfast,
        tags=["logging"],
    ),
    Case(
        id="02_approximate_portion",
        description="'leftover biryani, maybe two thirds of the box' -> fractional portion",
        turns=[Turn("leftover biryani, maybe two thirds of the box")],
        check=_check_approx_portion,
        tags=["logging", "portions"],
    ),
    Case(
        id="03_ambiguous_grazing",
        description="'skipped lunch but grazed all afternoon' -> asks, logs nothing",
        turns=[Turn("skipped lunch but grazed all afternoon")],
        check=_check_grazing,
        tags=["ambiguity"],
    ),
    Case(
        id="04_same_as_yesterday",
        description="'same as yesterday' -> replays yesterday's meals (needs memory)",
        turns=[Turn("same as yesterday")],
        setup=_setup_yesterday,
        check=_check_same_as_yesterday,
        tags=["memory", "differentiator"],
    ),
    Case(
        id="05_correction_updates_in_place",
        description="'actually that was 3 rotis not 2' -> UPDATES, never double-counts",
        turns=[Turn("had 2 rotis for lunch"), Turn("actually that was 3 rotis not 2")],
        check=_check_correction,
        tags=["corrections", "differentiator"],
    ),
    Case(
        id="06_protein_total",
        description="'how much protein have I had today?' -> accurate running total",
        turns=[Turn("how much protein have I had today?")],
        setup=_setup_known_day,
        check=_check_protein_query,
        tags=["totals"],
    ),
    Case(
        id="07_calorie_summary",
        description="'how am I doing on calories?' -> accurate daily summary",
        turns=[Turn("how am I doing on calories?")],
        setup=_setup_known_day,
        check=_check_calorie_query,
        tags=["totals"],
    ),
    Case(
        id="08_photo_only",
        description="[photo] -> routed to vision model, identified food logged",
        turns=[Turn("", image=SAMPLE_IMAGE)],
        check=_check_photo,
        tags=["vision"],
    ),
    Case(
        id="09_photo_with_caption",
        description="[photo] + 'half of this was my brother's' -> ONE meal, half portion",
        turns=[Turn("half of this was my brother's", image=SAMPLE_IMAGE)],
        setup=_setup_full_plate,
        check=_check_photo_half,
        tags=["vision", "differentiator"],
    ),
    Case(
        id="10_my_usual",
        description="'my usual' -> resolved from memory and logged",
        turns=[Turn("my usual")],
        setup=_setup_usual,
        check=_check_usual,
        tags=["memory", "differentiator"],
    ),
    Case(
        id="11_dietary_preference",
        description="'i'm vegetarian btw' -> stored in memory, not logged as a meal",
        turns=[Turn("i'm vegetarian btw")],
        check=_check_vegetarian,
        tags=["memory"],
    ),
]
