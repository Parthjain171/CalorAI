"""Run the eval set and report pass/fail plus latency.

    python eval/eval_runner.py                 # all cases
    python eval/eval_runner.py --case 05 09    # just these
    python eval/eval_runner.py --repeat 3      # more latency samples

Each case runs against a fresh user id, so cases cannot contaminate one another
while still sharing one database file — which is also a standing check that
per-user isolation actually holds.

Exit code is 0 only if every case passes, so this is usable as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.test_conversations import CASES, Case  # noqa: E402
from src.utils import latency  # noqa: E402
from src.utils.config import settings  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _run_case(agent: Any, case: Case, verbose: bool) -> Dict[str, Any]:
    """Run one case end to end and return its result record."""
    user_id = f"eval_{case.id}_{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    replies: List[str] = []
    context: Dict[str, Any] = {}

    try:
        if case.setup:
            context = case.setup(agent, user_id) or {}
        for turn in case.turns:
            reply = agent.chat(user_id, turn.text, image_path=turn.image)
            replies.append(reply)
            if verbose:
                label = turn.text or "(no text)"
                if turn.image:
                    label = f"[photo] {label}"
                print(f"    {DIM}> {label}{RESET}")
                print(f"    {DIM}< {reply}{RESET}")
        problems = case.check(user_id, replies, context)
    except Exception:
        problems = ["raised: " + traceback.format_exc(limit=3).strip().splitlines()[-1]]
        if verbose:
            traceback.print_exc()

    return {
        "case": case,
        "user_id": user_id,
        "passed": not problems,
        "problems": problems,
        "seconds": time.perf_counter() - started,
        "replies": replies,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CalorAI eval set")
    parser.add_argument("--case", nargs="*", help="substring(s) of case ids to run")
    parser.add_argument("--repeat", type=int, default=1, help="repeat for more latency samples")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every turn")
    parser.add_argument("--keep-db", action="store_true", help="do not wipe the database first")
    args = parser.parse_args(argv)

    selected = CASES
    if args.case:
        selected = [c for c in CASES if any(token in c.id for token in args.case)]
        if not selected:
            print(f"no cases matched {args.case}")
            return 2

    from src.agent import CalorAIAgent
    from src.db.schema import reset_database

    if not args.keep_db:
        reset_database()

    mode = "MOCK (scripted double)" if settings.mock else f"REAL ({settings.text_model})"
    print(f"CalorAI eval — {len(selected)} case(s) x{args.repeat} — mode: {mode}")
    if settings.mock:
        print(
            f"{YELLOW}Mock mode validates plumbing (tools, database, state "
            f"transitions), not answer quality.{RESET}"
        )
        print(f"{YELLOW}Latency here is framework overhead only — set "
              f"CALORAI_MOCK=0 with an API key for real numbers.{RESET}")
    print(f"vision: {settings.vision_model}\n")

    latency.reset()
    agent = CalorAIAgent()
    results: List[Dict[str, Any]] = []

    for iteration in range(args.repeat):
        if args.repeat > 1:
            print(f"{DIM}--- pass {iteration + 1}/{args.repeat} ---{RESET}")
        for case in selected:
            result = _run_case(agent, case, args.verbose)
            results.append(result)
            mark = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
            print(f"  {mark}  {case.id:<32} {result['seconds']:>6.2f}s  {case.description}")
            for problem in result["problems"]:
                print(f"        {RED}-> {problem}{RESET}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'=' * 72}")
    print(f"{passed}/{total} passed" + (f"  {RED}({total - passed} failed){RESET}" if passed < total else f"  {GREEN}all good{RESET}"))

    failed_ids = sorted({r["case"].id for r in results if not r["passed"]})
    if failed_ids:
        print("failing cases: " + ", ".join(failed_ids))

    print(f"\nLatency\n{'-' * 72}")
    print(latency.format_report(latency.spans()))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
