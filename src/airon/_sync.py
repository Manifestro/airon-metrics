"""Upload spooled runs: the work of ``airon sync``."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from ._spool import Spool, list_spools
from ._transport import AUTH, DROP, RETRY, Client, Result, send_split


class AuthError(Exception):
    """The server refused the key; there is no point trying other runs."""


@dataclass
class Outcome:
    run_id: str
    name: str
    events: int
    status: str  # "uploaded", "skipped" or "failed"
    message: str = ""


def _retrying(call: Callable[[], Result], attempts: int, min_delay: float) -> Result:
    result = call()
    for attempt in range(1, attempts):
        if result.outcome != RETRY:
            break
        delay = max(min_delay * 2 ** (attempt - 1), result.retry_after or 0.0)
        time.sleep(min(delay, 60.0))
        result = call()
    return result


def _is_active(meta: dict[str, object]) -> bool:
    """True if the process that spooled this run is still alive on this machine."""
    pid = meta.get("pid")
    if os.name != "posix" or not isinstance(pid, int) or pid == os.getpid():
        return False
    if meta.get("host") != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists, but belongs to someone else
    return True


def sync_run(
    client: Client, spool: Spool, force: bool = False, attempts: int = 4, min_delay: float = 1.0
) -> Outcome:
    meta = spool.read_meta() or {}
    create = meta.get("create")
    run_id = spool.directory.name
    if not isinstance(create, dict):
        return Outcome(run_id, run_id, 0, "failed", "meta.json is missing or unreadable")
    run_id = str(create.get("run_id", run_id))
    name = str(create.get("name", run_id))
    if not force and _is_active(meta):
        return Outcome(
            run_id, name, 0, "skipped", f"still running (pid {meta.get('pid')}); use --force"
        )

    # The run exists already if some events were delivered live, so ask to reopen it.
    result = _retrying(
        lambda: client.create_run({**create, "resume": "allow"}), attempts, min_delay
    )
    if not result.ok:
        return _failure(run_id, name, 0, result, "creating the run")

    uploaded = 0
    for segment in spool.segments():
        events = spool.read_segment(segment)
        if events:
            result = _retrying(partial(send_split, client, run_id, events), attempts, min_delay)
            if result.outcome == DROP:
                spool.remove_segment(segment)  # a refused batch will not be accepted later either
                continue
            if not result.ok:
                return _failure(run_id, name, uploaded, result, "sending events")
            uploaded += len(events)
        spool.remove_segment(segment)

    finish = spool.read_finish()
    if finish is not None:
        result = _retrying(lambda: client.finish_run(run_id, finish), attempts, min_delay)
        if not result.ok and result.outcome != DROP:
            return _failure(run_id, name, uploaded, result, "sending the finish request")
    spool.remove()
    return Outcome(run_id, name, uploaded, "uploaded")


def _failure(run_id: str, name: str, events: int, result: Result, doing: str) -> Outcome:
    if result.outcome == AUTH:
        raise AuthError(result.message)
    return Outcome(run_id, name, events, "failed", f"{doing}: {result.message or result.outcome}")


def sync_all(
    client: Client, root: Path, force: bool = False, attempts: int = 4, min_delay: float = 1.0
) -> list[Outcome]:
    return [sync_run(client, s, force, attempts, min_delay) for s in list_spools(root)]
