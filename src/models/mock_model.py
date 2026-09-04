"""A scripted, deterministic stand-in for the conversation model.

**This is a test double, not a feature.** It exists so the eval suite can
exercise the graph, the tools, and every database state transition without an
API key and without paying for tokens — CI can assert that a correction updates
in place, that totals stay consistent, and that memory survives a restart.

What it does *not* validate is the part a real model is actually for: intent
classification, portion inference, and natural phrasing. Passing the eval in
mock mode proves the plumbing is sound; it says nothing about answer quality.
Run with ``CALORAI_MOCK=0`` and a real key for that.

Enabled by ``CALORAI_MOCK=1``.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.tools.nutrition_data import ALIASES, SEED_NUTRITION

_WORD_QUANTITIES: Dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "couple": 2, "half": 0.5, "quarter": 0.25,
}
_FRACTION_PHRASES: List[Tuple[str, float]] = [
    ("two thirds", 0.67), ("two-thirds", 0.67), ("three quarters", 0.75),
    ("half of", 0.5), ("a third", 0.33), ("one third", 0.33), ("quarter of", 0.25),
]


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": _new_id(), "type": "tool_call"}


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _parse_foods(text: str) -> List[Dict[str, Any]]:
    """Pull ``{food, servings}`` pairs out of a message using the seed vocabulary."""
    lowered = text.lower()
    global_scale = 1.0
    for phrase, value in _FRACTION_PHRASES:
        if phrase in lowered:
            global_scale = value
            break

    found: List[Dict[str, Any]] = []
    consumed: List[Tuple[int, int]] = []
    vocabulary = sorted(
        set(SEED_NUTRITION) | set(ALIASES), key=len, reverse=True
    )
    for term in vocabulary:
        for match in re.finditer(
            rf"(?:(\d+(?:\.\d+)?|{'|'.join(_WORD_QUANTITIES)})\s+)?{re.escape(term)}e?s?\b",
            lowered,
        ):
            start, end = match.span()
            if any(s <= start < e or s < end <= e for s, e in consumed):
                continue
            consumed.append((start, end))
            raw_quantity = match.group(1)
            if raw_quantity is None:
                servings = 1.0
            elif raw_quantity in _WORD_QUANTITIES:
                servings = _WORD_QUANTITIES[raw_quantity]
            else:
                servings = float(raw_quantity)
            found.append({
                "food": term,
                "servings": round(servings * global_scale, 2),
                "at": start,
            })
    found.sort(key=lambda item: item["at"])
    return [{"food": f["food"], "servings": f["servings"]} for f in found]


def _meal_type(text: str) -> str:
    lowered = text.lower()
    for candidate in ("breakfast", "lunch", "dinner"):
        if candidate in lowered:
            return candidate
    if any(word in lowered for word in ("morning", "chai", "paratha", "idli", "poha")):
        return "breakfast"
    return "snack"


class ScriptedChatModel(BaseChatModel):
    """Rule-based model that emits the tool calls a real model would emit."""

    tool_names: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "calorai-scripted-mock"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "ScriptedChatModel":
        clone = ScriptedChatModel()
        clone.tool_names = [getattr(t, "name", str(t)) for t in tools]
        return clone

    # --- conversation bookkeeping -------------------------------------------

    @staticmethod
    def _turn_slice(messages: List[BaseMessage]) -> Tuple[str, List[BaseMessage]]:
        """Return the current user message plus everything produced since it."""
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return _text_of(messages[index]), messages[index + 1 :]
        return "", []

    @staticmethod
    def _tool_results(tail: Sequence[BaseMessage]) -> Dict[str, Any]:
        """Most recent result per tool name, parsed back into Python objects."""
        import ast

        results: Dict[str, Any] = {}
        for message in tail:
            if isinstance(message, ToolMessage):
                try:
                    results[message.name or "?"] = ast.literal_eval(str(message.content))
                except (ValueError, SyntaxError):
                    results[message.name or "?"] = str(message.content)
        return results

    # --- the script ----------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_text, tail = self._turn_slice(messages)
        # Vision output is injected as a system-role line; fold it into the text
        # the rules see so a photo and its caption resolve as one meal.
        vision_note = ""
        for message in messages:
            content = _text_of(message)
            if content.startswith("[VISION]"):
                vision_note = content
        combined = f"{vision_note} {user_text}".strip()
        results = self._tool_results(tail)
        rounds = sum(
            1 for m in tail if isinstance(m, AIMessage) and m.tool_calls
        )
        message = self._decide(combined, user_text, results, rounds)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _decide(
        self, combined: str, user_text: str, results: Dict[str, Any], rounds: int
    ) -> AIMessage:
        lowered = combined.lower()

        # 1. Corrections must never create a second row.
        if re.search(r"\b(actually|not \d|i meant|correction|make that|instead)\b", lowered):
            return self._correction(lowered, results, rounds)

        # 2. Questions about the day.
        if re.search(r"\b(how am i doing|how much|how many|what.*total|left|so far)\b", lowered):
            if rounds == 0:
                return AIMessage(content="", tool_calls=[_tool_call("get_daily_totals", {})])
            totals = results.get("get_daily_totals", {})
            return AIMessage(content=self._totals_reply(lowered, totals))

        # 3. Durable facts about the user.
        preference = self._preference(user_text)
        if preference and rounds == 0:
            return AIMessage(content="", tool_calls=[_tool_call("store_memory", preference)])
        if preference and rounds >= 1:
            return AIMessage(content="Got it, noted. I'll keep that in mind from now on.")

        # 4. Replay past meals.
        if "same as yesterday" in lowered:
            return self._replay_yesterday(results, rounds)

        # 5. "my usual" needs memory before it can be logged.
        if "usual" in lowered:
            return self._usual(results, rounds)

        # 6. Too vague to log — ask rather than invent.
        foods = _parse_foods(combined)
        if not foods:
            return AIMessage(content=self._clarify(lowered))

        # 7. The default path: price it, log it, confirm it.
        if rounds == 0:
            return AIMessage(
                content="", tool_calls=[_tool_call("lookup_nutrition", {"items": foods})]
            )
        if rounds == 1 and "lookup_nutrition" in results:
            total = results["lookup_nutrition"]["total"]
            source = "vision+text" if combined.startswith("[VISION]") else "text"
            name = ", ".join(
                f"{f['servings']:g} {f['food']}" for f in foods
            )
            return AIMessage(content="", tool_calls=[_tool_call("log_meal", {
                "meal_name": name, "description": user_text, "source": source,
                "meal_type": _meal_type(combined), **total,
            })])
        logged = results.get("log_meal", {})
        return AIMessage(content=self._logged_reply(logged))

    # --- individual flows ----------------------------------------------------

    @staticmethod
    def _corrected_items(
        meal_name: str, target: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Re-price the WHOLE meal with the corrected quantity substituted in.

        "2 rotis and dal" corrected to "3 rotis" must become 3 rotis *plus the
        dal* — re-pricing only the corrected food would silently drop the rest
        of the meal.
        """
        items = _parse_foods(meal_name) or []
        for item in items:
            if item["food"] == target["food"]:
                item["servings"] = target["servings"]
                return items
        return items + [target] if items else [target]

    def _correction(
        self, lowered: str, results: Dict[str, Any], rounds: int
    ) -> AIMessage:
        foods = _parse_foods(lowered)
        target = foods[0] if foods else {"food": "", "servings": 1.0}
        if rounds == 0:
            return AIMessage(content="", tool_calls=[_tool_call(
                "get_meals", {"period": "today", "name_contains": target["food"]}
            )])

        meals = results.get("get_meals", {}).get("meals", [])
        if not meals:
            return AIMessage(content="I couldn't find that meal to correct — what was it?")

        if rounds == 1:
            items = self._corrected_items(meals[0]["meal_name"], target)
            return AIMessage(content="", tool_calls=[_tool_call(
                "lookup_nutrition", {"items": items}
            )])
        if rounds == 2:
            items = results["lookup_nutrition"]["items"]
            total = results["lookup_nutrition"]["total"]
            name = ", ".join(f"{i['servings']:g} {i['matched_as']}" for i in items)
            return AIMessage(content="", tool_calls=[_tool_call("update_meal", {
                "meal_id": meals[0]["meal_id"], "meal_name": name, **total,
            })])

        updated = results.get("update_meal", {})
        totals = updated.get("daily_totals", {})
        return AIMessage(
            content=f"Fixed — updated to {updated.get('updated', {}).get('meal_name', 'that meal')}. "
            f"You're at {totals.get('calories', 0):g} cal today."
        )

    def _replay_yesterday(self, results: Dict[str, Any], rounds: int) -> AIMessage:
        if rounds == 0:
            return AIMessage(content="", tool_calls=[
                _tool_call("get_meals", {"period": "yesterday", "limit": 20})
            ])
        if rounds == 1:
            meals = results.get("get_meals", {}).get("meals", [])
            if not meals:
                return AIMessage(content="I don't have anything logged for yesterday yet.")
            return AIMessage(content="", tool_calls=[
                _tool_call("log_meal", {
                    "meal_name": m["meal_name"], "description": "same as yesterday",
                    "meal_type": m["meal_type"], "calories": m["calories"],
                    "protein": m["protein"], "carbs": m["carbs"], "fat": m["fat"],
                }) for m in meals
            ])
        totals = results.get("log_meal", {}).get("daily_totals", {})
        return AIMessage(
            content=f"Logged the same as yesterday. You're at {totals.get('calories', 0):g} cal today."
        )

    def _usual(self, results: Dict[str, Any], rounds: int) -> AIMessage:
        if rounds == 0:
            return AIMessage(content="", tool_calls=[
                _tool_call("recall_memory", {"query": "usual"})
            ])
        memories = results.get("recall_memory", {}).get("memories", [])
        usual = next((m for m in memories if "usual" in m["key"]), None)
        if usual is None:
            return AIMessage(content="I don't know your usual yet — what do you normally have?")
        if rounds == 1:
            return AIMessage(content="", tool_calls=[
                _tool_call("lookup_nutrition", {"items": _parse_foods(usual["value"])})
            ])
        if rounds == 2:
            total = results["lookup_nutrition"]["total"]
            return AIMessage(content="", tool_calls=[_tool_call("log_meal", {
                "meal_name": usual["value"], "description": "my usual",
                "meal_type": _meal_type(usual["key"]), **total,
            })])
        return AIMessage(content=self._logged_reply(results.get("log_meal", {})))

    # --- reply helpers -------------------------------------------------------

    @staticmethod
    def _preference(text: str) -> Optional[Dict[str, str]]:
        lowered = text.lower()
        if re.search(r"\b(i'?m|i am) (a )?(vegetarian|vegan|veg)\b", lowered):
            diet = "vegan" if "vegan" in lowered else "vegetarian"
            return {"key": "diet", "value": diet, "category": "preference"}
        goal = re.search(r"(targeting|aiming for|goal(?: is)?)\s*(\d+)\s*g?\s*(protein|calories)", lowered)
        if goal:
            return {
                "key": f"{goal.group(3)}_goal",
                "value": f"{goal.group(2)}{'g' if goal.group(3) == 'protein' else ''} {goal.group(3)} per day",
                "category": "goal",
            }
        usual = re.search(r"my usual (breakfast|lunch|dinner) is (.+)", lowered)
        if usual:
            return {
                "key": f"usual_{usual.group(1)}",
                "value": usual.group(2).strip(" ."),
                "category": "usual_meal",
            }
        if re.search(r"\bi (don'?t|do not) eat ([a-z ]+)", lowered):
            match = re.search(r"\bi (don'?t|do not) eat ([a-z ]+)", lowered)
            return {"key": "avoids", "value": match.group(2).strip(), "category": "preference"}
        return None

    @staticmethod
    def _clarify(lowered: str) -> str:
        if "graz" in lowered:
            return (
                "No worries. What did you graze on through the afternoon? "
                "Even rough amounts help — chai, biscuits, namkeen, anything."
            )
        if "skip" in lowered:
            return "Got it, skipping lunch. Anything else you had instead?"
        return "Happy to log that — what did you eat exactly?"

    @staticmethod
    def _totals_reply(lowered: str, totals: Dict[str, Any]) -> str:
        if "protein" in lowered:
            return f"You're at {totals.get('protein', 0):g}g protein today."
        return (
            f"Today so far: {totals.get('calories', 0):g} cal — "
            f"{totals.get('protein', 0):g}g protein, {totals.get('carbs', 0):g}g carbs, "
            f"{totals.get('fat', 0):g}g fat across {totals.get('meal_count', 0)} meals."
        )

    @staticmethod
    def _logged_reply(logged: Dict[str, Any]) -> str:
        meal = logged.get("logged", {})
        totals = logged.get("daily_totals", {})
        return (
            f"Logged {meal.get('meal_name', 'that')} — {meal.get('calories', 0):g} cal, "
            f"{meal.get('protein', 0):g}g protein. "
            f"That puts you at {totals.get('calories', 0):g} cal today."
        )
