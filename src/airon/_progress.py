"""Progress and metrics for any stage of a job, by printing one line Airon reads from the log.

A script that already shows a tqdm bar needs nothing: the agent sends the bar and Airon reads
it. These are for the rest::

    for path in airon.track(files, "transcribe", unit="files"):
        ...
    airon.metric(wer=0.142, rtf=0.31)

Outside a job they print nothing and `track` just hands the items through.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")
EVERY = 5.0
_WORD = re.compile(r"[^A-Za-z0-9_.\-/]+")


def _in_job() -> bool:
    return bool(os.environ.get("AIRON_JOB"))


def _say(line: str) -> None:
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _word(text: str) -> str:
    return _WORD.sub("-", str(text)).strip("-")[:80] or "progress"


def progress(name: str, done: float, total: float | None = None, unit: str = "it") -> None:
    """Say how far a part of the job is: `progress("resample", 400, 3000, "files")`."""
    if not _in_job():
        return
    line = f"::airon progress name={_word(name)} done={done:g} unit={_word(unit)}"
    if total is not None:
        line += f" total={total:g}"
    _say(line)


def metric(step: int | None = None, **values: Any) -> None:
    """Any numbers, in any stage: `metric(wer=0.142)`. They show on the run as `<stage>/<name>`.
    Without `step`, each value is the next point of its series."""
    if not _in_job():
        return
    parts = []
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parts.append(f"{_word(name)}={number:.6g}")
    if not parts:
        return
    if step is not None:
        parts.insert(0, f"step={int(step)}")
    _say("::airon metric " + " ".join(parts))


def track(
    items: Iterable[T],
    name: str = "progress",
    total: int | None = None,
    unit: str = "it",
    every: float = EVERY,
) -> Iterator[T]:
    """Hand the items through, and tell Airon how many are done every few seconds and at the end.
    `total` defaults to len(items) when it has one."""
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    done = 0
    last = 0.0
    progress(name, 0, total, unit)
    for item in items:
        yield item
        done += 1
        now = time.monotonic()
        if now - last >= every:
            last = now
            progress(name, done, total, unit)
    progress(name, done, total, unit)
