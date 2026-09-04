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

from src.models.text_model import get_chat_model
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

CORRECTIONS
- "actually that was 3 rotis not 2", "make that a large" — the user is fixing a
  meal that ALREADY EXISTS. Find it with get_meals (use name_contains), then
  call update_meal with the corrected values. NEVER call log_meal for a fix;
  that double-counts and is the single worst thing you can do here.

QUESTIONS ABOUT THE DAY
- "how am I doing", "how much protein so far" -> get_daily_totals, then answer
  with the actual numbers. Do not estimate from memory.
"""


class AgentState(TypedDict):
    """State carried through one turn of the graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    image_path: Optional[str]


def _system_prompt() -> str:
    now = local_now()
    return SYSTEM_PROMPT.format(
        today=local_date_str(),
        weekday=now.strftime("%A"),
        time=now.strftime("%H:%M"),
    )


def _prepare(state: AgentState) -> Dict[str, Any]:
    """Refresh the system prompt for this turn."""
    prompt = SystemMessage(content=_system_prompt())
    history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    return {"messages": [prompt] + history}


def _agent_node(state: AgentState) -> Dict[str, Any]:
    """Call the conversation model with tools bound."""
    model = get_chat_model(settings.text_model).bind_tools(ALL_TOOLS)
    return {"messages": [model.invoke(state["messages"])]}


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("prepare", _prepare)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("prepare")
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
