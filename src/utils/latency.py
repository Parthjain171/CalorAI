"""Latency instrumentation.

Every timed span is appended to a JSONL file (``CALORAI_LATENCY_LOG``) as well as
kept in memory, so percentiles survive the process and the numbers in the README
come from real runs rather than a single lucky invocation.

Spans are nested on purpose — a ``turn_image`` span contains a ``vision_model``
span and one or more ``text_model_call`` spans — which is what makes it possible
to say *where* a slow turn went, not just that it was slow.

Percentiles use nearest-rank on the sorted sample. With the sample sizes an eval
run produces (tens, not thousands), a p95 is a rough indicator; the report says
so rather than implying more precision than the data supports.
"""

from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.utils.config import settings


@dataclass
class Span:
    """One timed operation."""

    label: str
    seconds: float
    started_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


_spans: List[Span] = []
_enabled = True
# LangGraph runs nodes on worker threads, so record() is genuinely concurrent.
# Buffered text-mode appends from two threads can interleave and shred a line -
# observed as a record whose head was overwritten by the next writer, leaving
# only its tail. One lock around the append fixes it.
_write_lock = threading.Lock()


def set_enabled(enabled: bool) -> None:
    """Turn recording off (used when timing would just be noise)."""
    global _enabled
    _enabled = enabled


def record(label: str, seconds: float, **meta: Any) -> Span:
    """Record one span in memory and append it to the log file."""
    span = Span(label=label, seconds=seconds, started_at=time.time(), meta=meta)
    if not _enabled:
        return span
    line = json.dumps(asdict(span)) + "\n"
    # Single append under a lock, and no retry: a retry is unsafe because a
    # partially-landed write would be duplicated. A dropped line is the safer
    # failure mode, and load_spans() skips anything unparseable.
    #
    # Writes can still fail when data/ sits in a sync-client folder (OneDrive,
    # Dropbox) that momentarily locks a freshly created file. The in-memory
    # list is authoritative within a process; the JSONL is durable history.
    with _write_lock:
        _spans.append(span)
        try:
            path = settings.latency_log
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass  # Never let instrumentation break a turn.
    return span


@contextmanager
def measure(label: str, **meta: Any) -> Iterator[Dict[str, Any]]:
    """Time a block. The yielded dict can be updated with extra metadata."""
    extra: Dict[str, Any] = dict(meta)
    start = time.perf_counter()
    try:
        yield extra
    finally:
        record(label, time.perf_counter() - start, **extra)


def spans(label: Optional[str] = None) -> List[Span]:
    """In-memory spans, optionally filtered by label."""
    return [s for s in _spans if label is None or s.label == label]


def reset() -> None:
    """Clear in-memory spans (the log file is left alone)."""
    _spans.clear()


def load_spans(path: Optional[Path] = None) -> List[Span]:
    """Read spans back from the JSONL log."""
    target = path or settings.latency_log
    if not target.exists():
        return []
    out: List[Span] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Span(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` is 0-100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(source: Optional[Sequence[Span]] = None) -> Dict[str, Dict[str, float]]:
    """Per-label count, p50, p95, mean, min and max, in seconds."""
    data = source if source is not None else _spans
    grouped: Dict[str, List[float]] = {}
    for span in data:
        grouped.setdefault(span.label, []).append(span.seconds)

    return {
        label: {
            "count": len(values),
            "p50": round(percentile(values, 50), 3),
            "p95": round(percentile(values, 95), 3),
            "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
        for label, values in sorted(grouped.items())
    }


def format_report(source: Optional[Sequence[Span]] = None) -> str:
    """Render a fixed-width latency table."""
    stats = summarize(source)
    if not stats:
        return "No latency data recorded yet."

    header = f"{'span':<20} {'n':>4} {'p50':>8} {'p95':>8} {'mean':>8} {'max':>8}"
    lines = [header, "-" * len(header)]
    for label, values in stats.items():
        lines.append(
            f"{label:<20} {values['count']:>4} {values['p50']:>7.2f}s "
            f"{values['p95']:>7.2f}s {values['mean']:>7.2f}s {values['max']:>7.2f}s"
        )
    small = [label for label, v in stats.items() if v["count"] < 20]
    if small:
        lines.append("")
        lines.append(
            "note: p95 over fewer than 20 samples is indicative only "
            f"({', '.join(small)})."
        )
    return "\n".join(lines)
