"""The LangGraph agent.

    prepare -> agent -> (tools -> agent)* -> END

``prepare`` builds the per-turn system prompt. ``agent`` is the conversation
model with tools bound. ``tools`` executes whatever it asked for and loops back.
LangGraph (rather than an AgentExecutor) because the graph makes the control
flow explicit - which matters once a vision pre-pass and a memory pre-pass sit in
front of the model.

Conversation history within a session is held by a LangGraph checkpointer keyed
on ``user_id``. Everything that must outlive the process - meals, memories - 
lives in SQLite, not in the message log.
"""

from __future__ import annotations

import sys
import time
from typing import Annotated, Any, Dict, Iterator, List, Optional, TypedDict

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
from src.utils.latency import measure, record
from src.utils.user_context import user_scope

SYSTEM_PROMPT = """You are CalorAI, a calorie tracker people text over WhatsApp. \
Talk like a friend, not a form.

Today is {today} ({weekday}), {time}.

VOICE: one or two sentences, no lists, no lectures. Confirm what you logged \
with its calories, then the day's running total.

LOGGING
- lookup_nutrition ONCE with every food in the message, then log_meal ONCE.
  One eating occasion = one meal; a photo plus its caption is one meal.
  meal_name names the food ("chicken quinoa bowl"), never the user's sentence.
- Approximate portions become a servings fraction ("two thirds of the box"
  -> 0.67, "half" -> 0.5, "a couple" -> 2). Estimate and move on.

ASK VS LOG
- Foods named, however vaguely quantified -> log. No food named at all
  ("had some food", "grazed all afternoon") -> ask ONE short question that
  names examples. Never ask twice.
- Asking is a plain text reply. Only the listed tools exist; never invent one.
- The message after your question IS the answer: log it. Partial answers get
  logged with a stated assumption.

PHOTOS: a [VISION] note lists what was identified. The user's words override
it - "half of this was my brother's" means ONE meal at 0.5 servings, never a
photo meal plus a caption meal. If the note says confidence is LOW, confirm
first ("looks like rice and dal - is that right?").

CORRECTIONS: "actually that was 3 rotis not 2" fixes an EXISTING meal:
get_meals (name_contains) -> update_meal with new totals. NEVER log_meal a
fix; that double-counts.

TOTALS: "how am I doing" -> get_daily_totals and quote the real numbers.

MEMORY: durable facts (diet, goals, habits, what "my usual" means) ->
store_memory, silently, same turn. Never meals or moods. "same as yesterday"
-> get_meals(period="yesterday"), then ONE response containing a log_meal call
for EVERY returned meal, reusing its stored calories and macros, with NO
meal_date (they are eaten today), then reply with the new total. "my usual"
-> recall_memory.
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
    text model reads the photo and the caption in the same turn - which is what
    collapses "photo + 'half of this was my brother's'" into a single meal.
    """
    image_path = state.get("image_path")
    if not image_path:
        return {}

    caption = _latest_user_text(state["messages"])
    with measure("vision_model", model=settings.vision_model):
        analysis = analyze_image(image_path, caption=caption)
    note = format_vision_note(analysis)

    # Merge the note INTO the user's own message rather than appending a new
    # one. A separate message would become the "latest user message", hiding the
    # caption from anything that reads it - and the caption is exactly what
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
    wholesale - see :mod:`src.memory.manager` for the tiering and the cap.
    """
    memories = recall(state["user_id"], _latest_user_text(state["messages"]))
    prompt = _system_prompt()
    memory_block = format_for_prompt(memories)
    if memory_block:
        prompt = f"{prompt}\n{memory_block}\n"
    return {"system_prompt": prompt}


def _bounded_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """The slice of the thread the model actually sees.

    The checkpointer keeps the whole thread, and the first real-model run showed
    why that must not be what gets sent: every past tool call and its JSON
    result was replayed on every request, so calls grew slower turn by turn and
    a free-tier token budget was exhausted by turn three. Facts live in SQLite,
    not in the transcript, so only recent context is needed.

    Trimmed from the end, starting on a human message, never splitting a tool
    call from its result. Budget: ``CALORAI_MAX_HISTORY_TOKENS``.
    """
    try:
        from langchain_core.messages.utils import count_tokens_approximately, trim_messages

        return trim_messages(
            messages,
            max_tokens=settings.max_history_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
    except Exception:  # noqa: BLE001 - never let trimming break a turn
        return messages[-12:]


_INVALID_TOOL_MARKERS = ("tool_use_failed", "not in request.tools", "unknown tool")

_INVALID_TOOL_NUDGE = (
    "Your previous attempt called a tool that does not exist. Only the tools "
    "provided exist. If you want to ask the user something or simply reply, "
    "write it as plain text now."
)

_EMPTY_REPLY_NUDGE = (
    "You stopped without replying. Finish the task: if a meal still needs "
    "logging, call log_meal; otherwise answer the user in one or two sentences."
)


def _is_empty(response: Any) -> bool:
    """True when the model returned neither a tool call nor any text."""
    if getattr(response, "tool_calls", None):
        return False
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return not str(content).strip()


def _agent_node(state: AgentState) -> Dict[str, Any]:
    """Call the conversation model with tools bound.

    Some models (seen with gpt-oss on Groq) occasionally express "ask the user
    a question" as a call to an imaginary tool, which strict providers reject
    with a 400 before we ever see a message. One retry with an explicit nudge
    turns that into the plain-text question it should have been, instead of
    surfacing a provider error to the user.
    """
    model = get_chat_model(settings.text_model).bind_tools(ALL_TOOLS)
    history = _bounded_history(list(state["messages"]))
    conversation = [SystemMessage(content=state["system_prompt"])] + history
    with measure("text_model_call", model=settings.text_model) as span:
        span["history_messages"] = len(history)
        try:
            response = model.invoke(conversation)
        except Exception as exc:  # noqa: BLE001 - provider-specific error classes vary
            if not any(marker in str(exc).lower() for marker in _INVALID_TOOL_MARKERS):
                raise
            span["retried_invalid_tool"] = True
            response = model.invoke(conversation + [SystemMessage(content=_INVALID_TOOL_NUDGE)])
        # Seen with gpt-oss mid-way through "same as yesterday": a stop with no
        # text and no tool call, leaving a meal unlogged and the user with
        # silence. One nudge finishes the job.
        if _is_empty(response):
            span["retried_empty_reply"] = True
            response = model.invoke(conversation + [SystemMessage(content=_EMPTY_REPLY_NUDGE)])
        span["tool_calls"] = len(getattr(response, "tool_calls", []) or [])
    return {"messages": [response]}


def _sqlite_checkpointer() -> Any:
    """Conversation checkpointer backed by SQLite, so a thread survives restarts.

    With an in-memory saver, a clarifying question ("what did you graze on?")
    asked just before the process exits is forgotten by the next session, and
    the user's answer lands with no question to attach to. Persisting the
    checkpoint keeps the thread intact across restarts. It lives in its own
    file so the application schema in ``calorai.db`` stays clean.
    """
    import sqlite3

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        # Usually means the wrong interpreter (global python instead of the
        # project venv). Run in-memory rather than refuse to start; the only
        # loss is that threads will not survive a restart.
        print(
            "warning: langgraph-checkpoint-sqlite not installed in this interpreter; "
            "conversation threads will not persist across restarts. "
            "Activate the venv (.\\.venv\\Scripts\\activate) or pip install -r requirements.txt.",
            file=sys.stderr,
        )
        return MemorySaver()

    path = settings.checkpoint_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph writes checkpoints from worker threads.
    # SqliteSaver serialises its own access with an internal lock.
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the agent graph.

    Pass ``checkpointer=MemorySaver()`` for throwaway sessions (tests); the
    default persists conversation threads to SQLite.
    """
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

    return graph.compile(checkpointer=checkpointer or _sqlite_checkpointer())


class CalorAIAgent:
    """Thin session wrapper around the compiled graph."""

    def __init__(self, persistent: bool = True) -> None:
        self.graph = build_graph(None if persistent else MemorySaver())

    def chat(
        self, user_id: str, message: str, image_path: Optional[str] = None
    ) -> str:
        """Run one turn and return the assistant's reply text."""
        label = "turn_image" if image_path else "turn_text"
        with user_scope(user_id), measure(label, user=user_id):
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

    def stream_chat(
        self, user_id: str, message: str, image_path: Optional[str] = None
    ) -> Iterator[str]:
        """Run one turn, yielding reply text as it is generated.

        Tool-calling rounds produce no visible text, so only the final answer
        surfaces. Streaming does not make a turn faster, but it cuts the time
        until the user sees *something* - which is the latency they feel.
        """
        label = "turn_image" if image_path else "turn_text"
        first_token_at: Optional[float] = None
        start = time.perf_counter()

        with user_scope(user_id), measure(label, user=user_id, streamed=True) as span:
            for chunk, meta in self.graph.stream(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_id": user_id,
                    "image_path": image_path,
                    "system_prompt": "",
                },
                config={"configurable": {"thread_id": user_id}, "recursion_limit": 25},
                stream_mode="messages",
            ):
                if meta.get("langgraph_node") != "agent":
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                if not content:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    record("time_to_first_token", first_token_at - start, user=user_id)
                    span["ttft"] = round(first_token_at - start, 3)
                yield str(content)
