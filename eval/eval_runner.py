"""Run the eval set and report pass/fail plus latency.

    python eval/eval_runner.py                 # all cases
    python eval/eval_runner.py --case 05 09    # just these
    python eval/eval_runner.py --repeat 3      # more latency samples

Each case runs against a fresh user id, so cases cannot contaminate one another
while still sharing one database file - which is also a standing check that
per-user isolation actually holds.

Exit codes: 0 when every case passes, 1 when an assertion failed, 3 when at
least one case raised instead of finishing (a crash is not a verdict), 2 when
no case matched. Non-zero either way, so this is usable as a CI gate.
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

from src.utils.bootstrap import reexec_in_venv  # noqa: E402 - stdlib only

reexec_in_venv(__file__)

from eval.test_conversations import CASES, Case  # noqa: E402
from src.utils import latency  # noqa: E402
from src.utils.config import settings  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

EXIT_OK, EXIT_FAILED, EXIT_NO_MATCH, EXIT_RAISED = 0, 1, 2, 3


def _run_case(agent: Any, case: Case, verbose: bool, pace: float = 0.0) -> Dict[str, Any]:
    """Run one case end to end and return its result record.

    ``pace`` sleeps that many seconds before each turn. Free API tiers meter
    tokens per minute; without pacing, a multi-call turn arriving right after
    another one is throttled and the SDK's backoff wait gets recorded as model
    latency. Pacing measures inference; running unpaced measures the tier.
    """
    user_id = f"eval_{case.id}_{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    replies: List[str] = []
    context: Dict[str, Any] = {}

    try:
        if case.setup:
            context = case.setup(agent, user_id) or {}
        for turn in case.turns:
            if pace > 0:
                time.sleep(pace)
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
        # The last line of the formatted trace is LangGraph's "During task with
        # name 'agent'" wrapper, which hides the real cause (a missing module,
        # a 401, a bad key). Report the innermost exception instead.
        exc = sys.exc_info()[1]
        while exc is not None and exc.__cause__ is not None:
            exc = exc.__cause__
        problems = [f"raised: {type(exc).__name__}: {exc}"]
        raised = True
        if verbose:
            traceback.print_exc()
    else:
        raised = False

    return {
        "case": case,
        "user_id": user_id,
        "passed": not problems,
        "raised": raised,
        "problems": problems,
        "seconds": time.perf_counter() - started,
        "replies": replies,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the selected cases and return 0 only if every one of them passed."""
    parser = argparse.ArgumentParser(description="Run the CalorAI eval set")
    parser.add_argument("--case", nargs="*", help="substring(s) of case ids to run")
    parser.add_argument("--repeat", type=int, default=1, help="repeat for more latency samples")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every turn")
    parser.add_argument("--keep-db", action="store_true", help="do not wipe the database first")
    parser.add_argument(
        "--pace", type=float, default=0.0,
        help="seconds to sleep before each turn (stay under a free tier's tokens/minute)",
    )
    args = parser.parse_args(argv)

    selected = CASES
    if args.case:
        selected = [c for c in CASES if any(token in c.id for token in args.case)]
        if not selected:
            print(f"no cases matched {args.case}")
            return EXIT_NO_MATCH

    from src.agent import CalorAIAgent
    from src.db.schema import reset_database

    if not args.keep_db:
        reset_database()

    mode = "MOCK (scripted double)" if settings.mock else f"REAL ({settings.text_model})"
    print(f"CalorAI eval - {len(selected)} case(s) x{args.repeat} - mode: {mode}")
    if settings.mock:
        print(
            f"{YELLOW}Mock mode validates plumbing (tools, database, state "
            f"transitions), not answer quality.{RESET}"
        )
        print(f"{YELLOW}Latency here is framework overhead only - set "
              f"CALORAI_MOCK=0 with an API key for real numbers.{RESET}")
    print(f"vision: {settings.vision_model}\n")

    latency.reset()
    agent = CalorAIAgent()
    results: List[Dict[str, Any]] = []

    for iteration in range(args.repeat):
        if args.repeat > 1:
            print(f"{DIM}--- pass {iteration + 1}/{args.repeat} ---{RESET}")
        for case in selected:
            result = _run_case(agent, case, args.verbose, pace=args.pace)
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
    if passed == total:
        return EXIT_OK
    if any(r["raised"] for r in results):
        print(f"{YELLOW}at least one case raised instead of finishing - fix the "
              f"environment before reading these results{RESET}")
        return EXIT_RAISED
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
