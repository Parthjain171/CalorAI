"""The vision path: a different model, doing a different job.

Photos do not go to the conversation model. A dedicated vision model
(``VISION_MODEL``, default ``claude-sonnet-5``) looks at the plate and returns
*structured food identification only* — no chat, no logging, no tool calls. Its
JSON is then handed to the text model as context.

**The handoff**, concretely:

1. CLI receives an image path (plus an optional caption).
2. ``analyze_image`` base64-encodes it and asks the vision model for JSON:
   which foods, how many servings, and how confident it is.
3. The agent's ``vision`` node turns that JSON into one ``[VISION] ...`` message
   appended to the conversation, right after whatever the user typed.
4. The text model reads the caption and the vision note *together* and decides
   what to do — which is what makes "half of this was my brother's" resolve to
   ONE meal at 0.5 servings instead of a photo meal plus a caption meal.
5. Low confidence is passed through rather than hidden, so the text model can
   ask "looks like rice and dal — is that right?" instead of guessing.

Splitting it this way means the expensive model is used only for the thing it is
uniquely good at, and every downstream behaviour (portions, memory, corrections,
totals) keeps working exactly as it does for text.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from src.models.text_model import provider_for
from src.utils.config import settings

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

VISION_PROMPT = """Identify the food in this photo for a calorie tracker.

Reply with ONLY a JSON object, no prose:
{
  "foods": [{"food": "<simple name>", "servings": <number>}],
  "confidence": <0.0-1.0>,
  "description": "<short plain description of the plate>",
  "question": "<a short confirming question, or null if you are confident>"
}

Rules:
- Use simple, common food names ("dal", "rice", "roti"), not menu descriptions.
- servings counts natural portions: 3 rotis -> 3, one bowl of dal -> 1.
- Judge portions from the plate: a half-full bowl is 0.5.
- If you are not sure what a dish is, still give your best guess, set confidence
  below 0.6, and put a short confirming question in "question".
"""

# Returned in mock mode so the image path can be exercised without an API key.
_MOCK_ANALYSIS: Dict[str, Any] = {
    "foods": [
        {"food": "rice", "servings": 1.0},
        {"food": "dal", "servings": 1.0},
        {"food": "roti", "servings": 2.0},
    ],
    "confidence": 0.82,
    "description": "a plate of rice, dal and two rotis",
    "question": None,
}


def _encode(path: Path) -> tuple[str, str]:
    """Return ``(mime_type, base64_data)`` for an image on disk."""
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        raise ValueError(
            f"Unsupported image type {path.suffix!r}. Use jpg, png, gif or webp."
        )
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def _image_block(mime: str, data: str) -> Dict[str, Any]:
    """Providers disagree on the multimodal block shape; normalise here.

    Anthropic takes a native ``image`` block. Google (Gemini) and every
    OpenAI-compatible host accept the ``image_url`` data-URL form, so they share
    a branch.
    """
    if provider_for(settings.vision_model) == "anthropic":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data},
        }
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _coerce(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Defend against a model that returns a *nearly* right shape."""
    foods: List[Dict[str, Any]] = []
    for entry in payload.get("foods") or []:
        if isinstance(entry, str):
            foods.append({"food": entry, "servings": 1.0})
        elif isinstance(entry, dict) and entry.get("food"):
            try:
                servings = float(entry.get("servings", 1) or 1)
            except (TypeError, ValueError):
                servings = 1.0
            foods.append({"food": str(entry["food"]), "servings": servings})
    try:
        confidence = float(payload.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    question = payload.get("question")
    return {
        "foods": foods,
        "confidence": max(0.0, min(1.0, confidence)),
        "description": str(payload.get("description") or "").strip(),
        "question": str(question).strip() if question else None,
    }


def analyze_image(image_path: str, caption: str = "") -> Dict[str, Any]:
    """Run the vision model over one photo and return structured food data.

    Args:
        image_path: Path to a jpg/png/gif/webp file.
        caption: What the user typed alongside the photo. Passed as a hint only
            — portion arithmetic stays with the text model, which sees both.

    Returns:
        ``{"foods", "confidence", "description", "question", "model", "error"}``.
    """
    path = Path(image_path).expanduser()
    if not path.is_file():
        return {
            "foods": [], "confidence": 0.0, "description": "", "question": None,
            "model": settings.vision_model, "error": f"No image at {path}",
        }

    if settings.mock:
        return {**_MOCK_ANALYSIS, "model": "mock-vision", "error": None}

    try:
        from src.models.text_model import get_chat_model as _factory

        mime, data = _encode(path)
        prompt = VISION_PROMPT
        if caption:
            prompt += f'\nThe user said: "{caption}" — use it only as a hint to identify the food.'

        # Built directly rather than through get_chat_model so mock mode cannot
        # swap the vision model out from under the image path.
        from src.models.text_model import _build  # noqa: PLC0415

        model = _build(settings.vision_model, 0.0, 700)
        response = model.invoke([
            HumanMessage(content=[{"type": "text", "text": prompt}, _image_block(mime, data)])
        ])
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
        match = re.search(r"\{.*\}", str(raw), re.S)
        if not match:
            raise ValueError("vision model did not return JSON")
        result = _coerce(json.loads(match.group(0)))
        return {**result, "model": settings.vision_model, "error": None}
    except Exception as exc:  # noqa: BLE001 - surface as a question, not a crash
        return {
            "foods": [], "confidence": 0.0, "description": "", "question": None,
            "model": settings.vision_model, "error": str(exc),
        }


def format_vision_note(analysis: Dict[str, Any]) -> str:
    """Render the vision result as the message the text model actually reads."""
    if analysis.get("error"):
        return (
            "[VISION] The photo could not be analysed "
            f"({analysis['error']}). Ask the user what was in it."
        )
    if not analysis["foods"]:
        return "[VISION] Nothing food-like was identified. Ask the user what they ate."

    items = ", ".join(f"{f['servings']:g}x {f['food']}" for f in analysis["foods"])
    confidence = analysis["confidence"]
    # FOODS: is a canonical, single-mention line. The prose description is
    # deliberately not repeated as a food list - anything parsing this note
    # would otherwise count every item twice.
    note = (
        f"[VISION] Photo analysed by {analysis['model']} "
        f"(confidence {confidence:.2f}).\nFOODS: {items}"
    )
    if confidence < 0.6:
        note += (
            " Confidence is LOW — confirm with the user before logging"
            f"{': ' + analysis['question'] if analysis['question'] else '.'}"
        )
    else:
        note += (
            "\nTreat this and anything the user typed as ONE meal. If they"
            " described a different portion, scale these servings accordingly."
        )
    if analysis["description"]:
        note += f"\n(Plate looked like: {analysis['description']}.)"
    return note
