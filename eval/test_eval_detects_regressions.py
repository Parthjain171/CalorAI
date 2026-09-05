"""Does the eval actually catch the bugs it claims to catch?

An eval that passes is only meaningful if it can fail. This deliberately breaks
the two differentiator behaviours and asserts that the corresponding case turns
red - guarding against assertions that are silently vacuous.

    python eval/test_eval_detects_regressions.py

Runs against the scripted double (it patches that double's logic), so it needs
no API key: mock mode is forced here regardless of ``.env``, because sabotaging
the double while a real model answers proves nothing. A case that *raises*
(missing dependency, bad key) does not count as detected either - a crash is
not the assertion doing its job. Exit code 0 means every sabotage was detected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.bootstrap import reexec_in_venv  # noqa: E402 - stdlib only

reexec_in_venv(__file__)

# Before src.utils.config is imported anywhere, so settings.mock is True from
# the start; reload_settings() below covers the case where it already was.
os.environ["CALORAI_MOCK"] = "1"

from langchain_core.messages import AIMessage  # noqa: E402

from eval import eval_runner  # noqa: E402
from src.models.mock_model import (  # noqa: E402
    ScriptedChatModel,
    _parse_foods,
    _tool_call,
)
from src.utils.config import reload_settings  # noqa: E402

reload_settings()


def _correction_that_double_counts(self: Any, lowered: str, results: Any, rounds: int) -> AIMessage:
    """The classic bug: treat a correction as a brand-new meal."""
    if rounds == 0:
        return AIMessage(content="", tool_calls=[
            _tool_call("lookup_nutrition", {"items": _parse_foods(lowered)})
        ])
    if rounds == 1:
        return AIMessage(content="", tool_calls=[_tool_call("log_meal", {
            "meal_name": "3 rotis", "meal_type": "lunch",
            **results["lookup_nutrition"]["total"],
        })])
    return AIMessage(content="Logged.")


_REAL_VISION_FOODS = ScriptedChatModel._vision_foods


def _vision_ignoring_caption(vision: str, caption: str) -> List[dict]:
    """Log the whole plate, ignoring "half of this was my brother's"."""
    return _REAL_VISION_FOODS(vision, "")


SABOTAGES: List[Tuple[str, str, Callable[..., Any], str]] = [
    (
        "correction logs a second meal instead of updating",
        "_correction",
        _correction_that_double_counts,
        "05",
    ),
    (
        "photo caption ignored, full portion logged",
        "_vision_foods",
        staticmethod(_vision_ignoring_caption),
        "09",
    ),
]


def main() -> int:
    """Apply each sabotage in turn; return 0 only if the eval caught them all."""
    missed: List[str] = []
    for description, attribute, replacement, case_id in SABOTAGES:
        original = getattr(ScriptedChatModel, attribute)
        setattr(ScriptedChatModel, attribute, replacement)
        print(f"\n### sabotage: {description}  (case {case_id} must FAIL)")
        try:
            exit_code = eval_runner.main(["--case", case_id])
        finally:
            setattr(ScriptedChatModel, attribute, original)

        if exit_code == eval_runner.EXIT_OK:
            missed.append(f"{case_id}: {description}")
            print(f"### MISSED - case {case_id} still passed, the assertion is too weak")
        elif exit_code == eval_runner.EXIT_FAILED:
            print("### detected")
        else:
            missed.append(f"{case_id}: {description} (case raised or did not run, exit {exit_code})")
            print(f"### INCONCLUSIVE - case {case_id} raised or did not run; nothing was tested")

    print(f"\n{'=' * 72}")
    if missed:
        print(f"{len(missed)} sabotage(s) NOT detected:")
        for item in missed:
            print(f" - {item}")
        return 1
    print(f"all {len(SABOTAGES)} sabotages detected - the eval can fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
