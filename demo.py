"""Walk through all 11 required conversations, in order, with visible state.

    python demo.py            # real model (needs a key in .env)
    CALORAI_MOCK=1 python demo.py

Three of the eleven cases only make sense with history behind them: "same as
yesterday" needs a yesterday, "actually that was 3 rotis" needs rotis, and "my
usual" needs a stored usual. Run cold, those three correctly reply "I don't
have that yet" - which looks like a failure to someone reading down the list.
This script seeds exactly that history first, then runs the list top to bottom
and prints what the database holds after every turn, so each behaviour can be
seen rather than inferred.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from src.agent import CalorAIAgent
from src.db.queries import daily_totals, insert_meal, list_memories, list_meals
from src.db.schema import connection, reset_database
from src.memory.manager import remember
from src.utils.config import local_date_str, settings

USER = "demo"
IMAGE = "assets/sample_plate.png"

SCRIPT = [
    ("had 2 parathas and chai for breakfast", None),
    ("leftover biryani, maybe two thirds of the box", None),
    ("skipped lunch but grazed all afternoon", None),
    ("chai and 3 biscuits", None),                       # answers the question above
    ("same as yesterday", None),
    ("had 2 rotis and dal", None),                        # gives the correction a target
    ("actually that was 3 rotis not 2", None),
    ("how much protein have I had today?", None),
    ("how am I doing on calories?", None),
    ("", IMAGE),
    ("half of this was my brother's", IMAGE),
    ("my usual", None),
    ("i'm vegetarian btw", None),
]


def seed_history() -> None:
    """Yesterday's meals and a stored usual, so cases 4 and 10 have something to use."""
    reset_database()
    yesterday = local_date_str(-1)
    for name, cal, pro, carb, fat, kind in [
        ("2 idli and sambar", 226, 9.0, 39.0, 3.6, "breakfast"),
        ("rajma chawal", 415, 16.2, 75.0, 4.4, "lunch"),
    ]:
        meal = insert_meal(USER, name, cal, pro, carb, fat, meal_type=kind)
        with connection() as conn:
            conn.execute("UPDATE meals SET meal_date = ? WHERE id = ?", (yesterday, meal["id"]))
            conn.commit()
    remember(USER, "usual_breakfast", "2 idli and sambar", "usual_meal")
    remember(USER, "protein_goal", "140g protein per day", "goal")


def show_state() -> None:
    """Print today's rows and totals straight from SQLite, bypassing the agent."""
    today = local_date_str()
    rows = list_meals(USER, start_date=today, end_date=today)
    totals = daily_totals(USER)
    for meal in reversed(rows):
        print(f"      #{meal['id']:<3} {meal['meal_name']:<34} {meal['calories']:>7g} cal  "
              f"P{meal['protein']:g}  [{meal['source']}]")
    print(f"      today: {totals['calories']:g} cal | P {totals['protein']:g}g | "
          f"C {totals['carbs']:g}g | F {totals['fat']:g}g | {totals['meal_count']} meals")


def run(text: str, image: Optional[str], agent: CalorAIAgent) -> None:
    """Send one turn, print the reply, then print what SQLite actually holds."""
    label = f"[photo] {text}".strip() if image else text
    print(f"\n  you:     {label}")
    reply = agent.chat(USER, text, image_path=image)
    print(f"  calorai: {reply}")
    show_state()


def main() -> None:
    """Seed, then run the script."""
    mode = "MOCK (scripted double)" if settings.mock else f"REAL ({settings.text_model})"
    print(f"CalorAI demo - mode: {mode}\n")
    seed_history()
    print("  seeded: 2 meals dated yesterday, usual_breakfast + protein_goal in memory")

    agent = CalorAIAgent(persistent=False)
    for text, image in SCRIPT:
        run(text, image, agent)

    print("\n  memories now:")
    for memory in list_memories(USER):
        print(f"      [{memory['category']:<10}] {memory['key']:<18} {memory['value']}")

    raw = sqlite3.connect(str(settings.db_path))
    count = raw.execute(
        "SELECT COUNT(*) FROM meals WHERE user_id = ? AND meal_date = ?",
        (USER, local_date_str()),
    ).fetchone()[0]
    raw.close()
    print(f"\n  direct SQL: {count} meal rows for today")


if __name__ == "__main__":
    main()
