"""CalorAI command-line chat.

    python cli.py                       # start chatting as the default user
    python cli.py --user parth          # a separate, isolated meal log
    python cli.py --image plate.jpg     # send one photo and exit
    python cli.py --latency             # print the latency report and exit

In-chat commands: /img, /totals, /meals, /memories, /latency, /reset, /help, /quit
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from typing import List, Optional

from src.agent import CalorAIAgent
from src.db.queries import daily_totals, list_meals, list_memories
from src.db.schema import get_connection, reset_database
from src.utils.config import settings
from src.utils.latency import format_report, load_spans

BANNER = """CalorAI — text me what you ate.
  "had 2 parathas and chai"     "how am I doing today?"
  /img <path> [caption]         send a photo
  /help                         all commands
"""

HELP = """Commands:
  /img <path> [caption]   log from a photo (quote captions with spaces)
  /totals [yesterday]     daily totals straight from the database
  /meals [yesterday]      list logged meals
  /memories               what the agent remembers about you
  /latency                p50/p95 for this session and all past runs
  /reset                  wipe ALL meals and memories (asks first)
  /quit                   exit
"""


def _print_totals(user_id: str, period: str = "today") -> None:
    from src.utils.config import local_date_str

    day = local_date_str(-1 if period == "yesterday" else 0)
    totals = daily_totals(user_id, day)
    print(
        f"  {totals['date']}: {totals['calories']:g} cal | "
        f"P {totals['protein']:g}g | C {totals['carbs']:g}g | F {totals['fat']:g}g "
        f"({totals['meal_count']} meals)"
    )


def _print_meals(user_id: str, period: str = "today") -> None:
    from src.utils.config import local_date_str

    day = local_date_str(-1 if period == "yesterday" else 0)
    meals = list_meals(user_id, start_date=day, end_date=day)
    if not meals:
        print(f"  nothing logged for {day}.")
        return
    for meal in reversed(meals):
        print(
            f"  #{meal['id']:<4} {meal['timestamp'][11:16]}  {meal['meal_name']:<34} "
            f"{meal['calories']:>6g} cal  P{meal['protein']:g}"
        )


def _print_memories(user_id: str) -> None:
    memories = list_memories(user_id)
    if not memories:
        print("  nothing remembered yet.")
        return
    for memory in memories:
        print(f"  [{memory['category']:<10}] {memory['key']:<18} {memory['value']}")


def _send(agent: CalorAIAgent, user_id: str, text: str, image: Optional[str], stream: bool) -> None:
    """Run one turn and print the reply, with the wall-clock time it took."""
    start = time.perf_counter()
    try:
        if stream:
            print("calorai: ", end="", flush=True)
            printed = False
            for piece in agent.stream_chat(user_id, text, image_path=image):
                print(piece, end="", flush=True)
                printed = True
            if not printed:
                print("(no reply)", end="")
            print()
        else:
            print(f"calorai: {agent.chat(user_id, text, image_path=image)}")
    except Exception as exc:  # noqa: BLE001 - a CLI should not traceback at users
        print(f"calorai: something went wrong — {exc}")
    print(f"         [{time.perf_counter() - start:.2f}s]")


def _handle_command(
    agent: CalorAIAgent, user_id: str, line: str, stream: bool
) -> bool:
    """Handle a /command. Returns False when the user asked to quit."""
    parts = shlex.split(line)
    command, args = parts[0].lower(), parts[1:]

    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        print(HELP)
    elif command == "/img":
        if not args:
            print("  usage: /img <path> [caption]")
        else:
            _send(agent, user_id, " ".join(args[1:]), args[0], stream)
    elif command == "/totals":
        _print_totals(user_id, args[0] if args else "today")
    elif command == "/meals":
        _print_meals(user_id, args[0] if args else "today")
    elif command == "/memories":
        _print_memories(user_id)
    elif command == "/latency":
        print(format_report(load_spans()))
    elif command == "/reset":
        if input("  wipe all meals and memories? [y/N] ").strip().lower() == "y":
            reset_database()
            print("  database cleared.")
    else:
        print(f"  unknown command {command}. /help for the list.")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CalorAI conversational meal logger")
    parser.add_argument("--user", default="default", help="user id (isolates meal logs)")
    parser.add_argument("--image", help="send one photo, print the reply, exit")
    parser.add_argument("--message", "-m", help="send one message, print the reply, exit")
    parser.add_argument("--latency", action="store_true", help="print latency report and exit")
    parser.add_argument("--no-stream", action="store_true", help="disable streaming output")
    args = parser.parse_args(argv)

    if args.latency:
        print(format_report(load_spans()))
        return 0

    get_connection()  # create the database up front so errors surface early
    stream = not args.no_stream
    agent = CalorAIAgent()

    if args.image or args.message:
        _send(agent, args.user, args.message or "", args.image, stream)
        return 0

    print(BANNER)
    print(f"user: {args.user} | text: {settings.text_model} | vision: {settings.vision_model}")
    if settings.mock:
        print("MOCK MODE — scripted test double, not a real model.")
    print()

    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(agent, args.user, line, stream):
                break
            continue
        _send(agent, args.user, line, None, stream)

    print("bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
