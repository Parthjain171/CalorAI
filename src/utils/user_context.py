"""Per-turn user scoping.

Tools need to know *whose* meals they are touching, but ``user_id`` is not
something the model should be able to choose - letting the LLM pass it invites
cross-user leakage and wastes tokens on an argument it can only get wrong. It is
carried out-of-band in a :class:`contextvars.ContextVar` that the agent sets
before each turn, so the tool schemas the model sees stay about food.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_current_user: ContextVar[str] = ContextVar("calorai_user_id", default="default")


def get_user_id() -> str:
    """The user whose data the currently-running tool should touch."""
    return _current_user.get()


@contextmanager
def user_scope(user_id: str) -> Iterator[str]:
    """Bind ``user_id`` for the duration of one agent turn."""
    token = _current_user.set(user_id)
    try:
        yield user_id
    finally:
        _current_user.reset(token)
