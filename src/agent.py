"""The LangGraph agent.

    prepare -> agent -> (tools -> agent)* -> END

``prepare`` builds the per-turn system prompt. ``agent`` is the conversation
model with tools bound. ``tools`` executes whatever it asked for and loops back.
LangGraph (rather than an AgentExecutor) because the graph makes the control
flow explicit — which matters once a vision pre-pass and a memory pre-pass sit in
front of the model.

Conversation history within a session is held by a LangGraph checkpointer keyed
on ``user_id``. Everything that must outlive the process — meals, memories —
lives in SQLite, not in the message log.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.memory.manager import format_for_prompt, recall
from src.models.text_model import get_chat_model
from src.models.vision_model import analyze_image, format_vision_note
from src.tools import ALL_TOOLS
from src.utils.config import local_date_str, local_now, settings
from src.utils.user_context import user_scope

SYSTEM_PROMPT = """You are CalorAI, a calorie tracker people talk to over \
WhatsApp. You are texting a friend, not filling in a form.

Today is {today} ({weekday}), local time {time}.

VOICE
- Short. One or two sentences. No bullet points, no headers, no emoji spam.
- Confirm what you logged with the calorie number, then the running day total.
- Never lecture about health, never moralise about what they ate.

LOGGING
- Call lookup_nutrition ONCE with every food in the message, then log_meal once.
- One eating occasion is ONE meal. "2 parathas and chai" is a single meal row,
  not two. A photo plus a caption about that photo is also a single meal.
- Approximate portions are normal. Convert them to a servings fraction:
  "two thirds of the box" -> 0.67, "half of this" -> 0.5, "a couple" -> 2.
  Estimate and move on; do not interrogate people about grams.

WHEN TO ASK VS WHEN TO LOG
- Enough to log: any message naming actual foods, even vaguely quantified.
  "had 2 parathas and chai" -> log it. "leftover biryani, maybe two thirds of
  the box" -> log it at 0.67 servings.
- Not enough: no food is named at all. "had some food", "ate out", "grazed all
  afternoon" -> ask ONE short question naming examples to make answering easy.
- Ask at most one clarifying question, then work with whatever you get back.
  Over-asking is worse than a slightly wrong estimate.
- After you ask, the next message IS the answer. Log it as the meal you asked
  about and do not ask a second time — if they say "chai and a few biscuits",
  that is enough. Never leave a clarification hanging unlogged.
- Partial answers still get logged. "some biscuits" after you asked is fine;
  assume a sensible number and say what you assumed.

PHOTOS
- A photo arrives already analysed, as a [VISION] message listing what was
  identified. Trust it, but the user's own words override it: if the vision note
  says 1x biryani and they typed "half of this was my brother's", that is ONE
  meal at 0.5 servings — never a photo meal plus a caption meal.
- If the [VISION] note says confidence is LOW, ask a short confirming question
  ("looks like rice and dal — is that right?") before logging.

CORRECTIONS
- "actually that was 3 rotis not 2", "make that a large" — the user is fixing a
  meal that ALREADY EXISTS. Find it with get_meals (use name_contains), then
  call update_meal with the corrected values. NEVER call log_meal for a fix;
  that double-counts and is the single worst thing you can do here.

QUESTIONS ABOUT THE DAY
- "how am I doing", "how much protein so far" -> get_daily_totals, then answer
  with the actual numbers. Do not estimate from memory.

MEMORY
- When someone tells you something durable about themselves — a diet, a goal, a
  standing habit, what "my usual" means — call store_memory. Do it silently in
  the same turn; do not make a ceremony of it.
- Do not store individual meals, moods, or one-off remarks.
- "same as yesterday" -> get_meals(period="yesterday"), then log those meals
  again for today. "my usual" -> recall_memory to find what it refers to.
"""


class AgentState(TypedDict):
    """State carried through one turn of the graph.

    ``system_prompt`` is held as a plain string rather than a message in
    ``messages``. The ``add_messages`` reducer appends, so a freshly built
    SystemMessage would stack up a new copy on every turn inside the
    checkpointer; keeping it out of the message list means it is rebuilt each
    turn and prepended once at call time.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    image_path: Optional[str]
    system_prompt: str


def _system_prompt() -> str:
    now = local_now()
    return SYSTEM_PROMPT.format(
        today=local_date_str(),
        weekday=now.strftime("%A"),
        time=now.strftime("%H:%M"),
    )


def _latest_user_text(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return str(content)
    return ""


def _vision(state: AgentState) -> Dict[str, Any]:
    """Run the vision model when a photo is attached, and only then.

    Its structured output is appended as one ``[VISION] ...`` message so the
    text model reads the photo and the caption in the same turn — which is what
    collapses "photo + 'half of this was my brother's'" into a single meal.
    """
    image_path = state.get("image_path")
    if not image_path:
        return {}

    caption = _latest_user_text(state["messages"])
    note = format_vision_note(analyze_image(image_path, caption=caption))

    # Merge the note INTO the user's own message rather than appending a new
    # one. A separate message would become the "latest user message", hiding the
    # caption from anything that reads it — and the caption is exactly what
    # turns "1x biryani" into half a portion. Reusing the message id makes
    # add_messages replace in place instead of appending.
    original = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    merged = f"{caption}\n\n{note}" if caption else note
    if original is None:
        return {"messages": [HumanMessage(content=merged)]}
    return {"messages": [HumanMessage(content=merged, id=original.id)]}


def _prepare(state: AgentState) -> Dict[str, Any]:
    """Rebuild the system prompt for this turn, with relevant memories injected.

    Memory is selected per turn against the incoming message rather than dumped
    wholesale — see :mod:`src.memory.manager` for the tiering and the cap.
    """
    memories = recall(state["user_id"], _latest_user_text(state["messages"]))
    prompt = _system_prompt()
    memory_block = format_for_prompt(memories)
    if memory_block:
        prompt = f"{prompt}\n{memory_block}\n"
    return {"system_prompt": prompt}


def _agent_node(state: AgentState) -> Dict[str, Any]:
    """Call the conversation model with tools bound."""
    model = get_chat_model(settings.text_model).bind_tools(ALL_TOOLS)
    conversation = [SystemMessage(content=state["system_prompt"])] + list(state["messages"])
    return {"messages": [model.invoke(conversation)]}


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("vision", _vision)
    graph.add_node("prepare", _prepare)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # vision no-ops on text-only turns, so the text path pays nothing for it.
    graph.set_entry_point("vision")
    graph.add_edge("vision", "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or MemorySaver())


class CalorAIAgent:
    """Thin session wrapper around the compiled graph."""

    def __init__(self) -> None:
        self.graph = build_graph()

    def chat(
        self, user_id: str, message: str, image_path: Optional[str] = None
    ) -> str:
        """Run one turn and return the assistant's reply text."""
        with user_scope(user_id):
            result = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_id": user_id,
                    "image_path": image_path,
                    "system_prompt": "",
                },
                config={"configurable": {"thread_id": user_id}, "recursion_limit": 25},
            )
        for message_obj in reversed(result["messages"]):
            if isinstance(message_obj, AIMessage) and not message_obj.tool_calls:
                content = message_obj.content
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    ).strip()
                return str(content).strip()
        return ""
