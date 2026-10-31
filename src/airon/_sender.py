"""The background thread that moves events from memory to the server, or to disk.

``Sender.enqueue`` is the only thing the training loop touches: an append to a bounded deque.
Everything slow (HTTP, retries, disk) happens on the sender thread, in this order of fallbacks:
send to the server, retry with backoff, write to the spool, upload the spool when the server is
reachable again. Events stay in order because while anything is on disk, new events go to disk too.
"""

from __future__ import annotations

import os
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from . import _log
from ._spool import Spool
from ._transport import AUTH, DROP, FATAL, OK, RETRY, Client, Result, send_split
from ._version import __version__

MAX_RETRY_AFTER = 300.0


@dataclass
class Tuning:
    flush_interval: float = 2.0
    heartbeat_interval: float = 30.0
    batch_size: int = 200
    max_queue: int = 100_000
    retry_min_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_attempts: int = 4  # per batch, before it is written to the spool
    shutdown_timeout: float = 10.0


@dataclass
class Stats:
    sent: int = 0
    spooled: int = 0
    dropped: int = 0
    rejected: int = 0


class Sender:
    def __init__(
        self,
        run_id: str,
        spool: Spool,
        client: Client | None,
        base_payload: dict[str, Any],
        tuning: Tuning,
        build_env: Callable[[], dict[str, Any]],
        on_created: Callable[[dict[str, Any]], None],
        extras: Callable[[], list[dict[str, Any]]],
    ):
        self.stats = Stats()
        self.t = tuning
        self._id = run_id
        self._spool = spool
        self._client = client
        self._base_payload = base_payload
        self._payload: dict[str, Any] | None = None
        self._build_env = build_env
        self._on_created = on_created
        self._extras = extras

        self._q: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._interrupt = threading.Event()
        self._seq = spool.max_seq()
        self._inflight: list[dict[str, Any]] | None = None

        self._created = False
        self._spooling = client is None  # offline runs only ever write to the spool
        self._dead = False  # the server will not take this run as configured
        self._had_outage = False
        self._failed_probes = 0
        self._next_probe = 0.0
        self._last_sent = time.monotonic()

        self._closing = False
        self._abandoned = False
        self._deadline = float("inf")
        self._finish_req: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._main, name="airon-sender", daemon=True)

    # ---- called from the training thread -------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        self._wake.set()  # create the run right away instead of after the first flush interval

    def enqueue(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._q.append(event)
            size = len(self._q)
        if size >= self.t.batch_size:
            self._wake.set()
        if size > self.t.max_queue:
            self._shed()

    @property
    def idle(self) -> bool:
        return not self._q and self._inflight is None

    def flush(self, timeout: float) -> None:
        """Wait until everything queued so far has left memory (sent or written to disk)."""
        end = time.monotonic() + timeout
        self._wake.set()
        while not self.idle and time.monotonic() < end and self._thread.is_alive():
            time.sleep(0.01)

    def close(self, finish: dict[str, Any], timeout: float | None = None) -> None:
        """Deliver what is left and the finish request, giving up after ``timeout`` seconds."""
        timeout = self.t.shutdown_timeout if timeout is None else timeout
        self._finish_req = finish
        self._deadline = time.monotonic() + timeout
        self._closing = True
        self._interrupt.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
        if self._thread.is_alive() or self._finish_req is not None:
            self._salvage()

    def _salvage(self) -> None:
        """The sender ran out of time (or never ran): put whatever is left on disk."""
        self._abandoned = True
        self._interrupt.set()
        with self._lock:
            pending = list(self._q)
            self._q.clear()
            batch = list(self._inflight or [])
            for event in pending:
                self._seq += 1
                event["seq"] = self._seq
        self._spool_batch(batch + pending)
        if self._finish_req is not None:
            self._spool.write_finish(self._finish_req)

    def _shed(self) -> None:
        """The queue is full: drop the oldest system events, then metrics, then log lines."""
        with self._lock:
            if len(self._q) <= self.t.max_queue:
                return
            target = len(self._q) - self.t.max_queue + self.t.max_queue // 10
            dropped = 0
            for kind in ("system", "metrics", "log", None):
                if dropped >= target:
                    break
                kept: deque[dict[str, Any]] = deque()
                for event in self._q:
                    if dropped < target and (kind is None or kind in event):
                        dropped += 1
                    else:
                        kept.append(event)
                self._q = kept
            self.stats.dropped += dropped
        _log.warn_once(
            "overflow",
            f"the event queue is full ({self.t.max_queue} events); dropping the oldest hardware "
            "samples and metrics. The server or the disk is too slow to keep up.",
        )

    # ---- the sender thread ---------------------------------------------------------------

    def _main(self) -> None:
        tick = max(0.01, min(self.t.flush_interval, self.t.heartbeat_interval))
        while True:
            self._wake.wait(tick if not self._closing else 0.05)
            self._wake.clear()
            if self._abandoned:
                return
            try:
                if self._cycle():
                    return
            except Exception as exc:  # the thread must outlive any bug in the code below
                _log.warn_once(
                    "internal", f"internal error in the sender: {type(exc).__name__}: {exc}"
                )
                if self._closing:
                    return

    def _cycle(self) -> bool:
        self._prepare()
        for event in self._extras():
            self.enqueue(event)
        while self._take_and_deliver():
            pass
        self._replay(force=False)
        self._heartbeat()
        return self._closing and self._finalize()

    def _prepare(self) -> None:
        if self._payload is not None:
            return
        payload = dict(self._base_payload)
        try:
            payload["env"] = self._build_env()
        except Exception:
            pass
        self._payload = payload
        if self._client is None:
            self._spool.write_meta(self._meta())
            return
        verdict, result = self._attempt(lambda: Result(OK), "the run", attempts=1)
        if verdict != "ok":
            self._start_spooling(result)

    def _meta(self) -> dict[str, Any]:
        return {
            "create": self._payload or self._base_payload,
            "pid": os.getpid(),
            "host": _hostname(),
            "sdk": __version__,
        }

    def _take_and_deliver(self) -> bool:
        with self._lock:
            if not self._q:
                return False
            batch = [self._q.popleft() for _ in range(min(len(self._q), self.t.batch_size))]
            for event in batch:
                self._seq += 1
                event["seq"] = self._seq
            self._inflight = batch
        try:
            if self._spooling:
                self._spool_batch(batch)
            else:
                self._send_batch(batch)
        finally:
            self._inflight = None
        return True

    def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        client = self._client
        assert client is not None
        verdict, result = self._attempt(lambda: send_split(client, self._id, batch), "events")
        if verdict == "ok":
            self._record_ack(result, len(batch))
        elif verdict == "drop":
            self.stats.dropped += len(batch)
        else:
            self._start_spooling(result)
            self._spool_batch(batch)

    def _record_ack(self, result: Result, count: int) -> None:
        self.stats.sent += count
        self._last_sent = time.monotonic()
        rejected = result.data.get("rejected")
        if isinstance(rejected, list) and rejected:
            self.stats.rejected += len(rejected)
            first = rejected[0] if isinstance(rejected[0], dict) else {}
            _log.warn_once(
                "rejected",
                f"the server refused {len(rejected)} event(s), for example seq {first.get('seq')}: "
                f"{first.get('reason')}. Rejected events are not resent.",
            )

    def _spool_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        if self._spool.append(batch, self._meta()):
            self.stats.spooled += len(batch)
            return
        self.stats.dropped += len(batch)
        _log.warn_once(
            "spool",
            f"cannot write the offline spool in {self._spool.directory} ({self._spool.last_error}); "
            "events that cannot be sent are lost",
        )

    # ---- retry policy ----------------------------------------------------------------------

    def _attempt(
        self, call: Callable[[], Result], what: str, attempts: int | None = None
    ) -> tuple[str, Result]:
        """Run ``call`` under the retry rules. Verdict is "ok", "drop" or "spool"."""
        limit = attempts or self.t.retry_attempts
        tries = 0
        while True:
            if self._abandoned:
                return "spool", Result(RETRY)
            result = self._ensure_created()
            if result is None:
                result = call()
            if result.ok:
                return "ok", result
            verdict = self._judge(result, what)
            if verdict != "retry":
                return verdict, result
            tries += 1
            if tries >= limit or (self._closing and time.monotonic() >= self._deadline):
                return "spool", result
            self._sleep(self._backoff(tries, result.retry_after))

    def _judge(self, result: Result, what: str) -> str:
        if result.outcome == AUTH:
            self._give_up(
                "auth",
                f"the server rejected the API key ({result.message}). Nothing more is sent; data "
                "is saved to disk. Fix the key (`airon doctor`), then run `airon sync`.",
            )
            return "spool"
        if result.outcome == FATAL:
            self._give_up("fatal", self._fatal_text(result))
            return "spool"
        if result.outcome == DROP:
            _log.warn_once(
                f"drop-{result.status}",
                f"the server refused the {what} ({result.message}); not retrying, dropped",
            )
            return "drop"
        return "retry"

    def _fatal_text(self, result: Result) -> str:
        reason = {
            409: f"run {self._id} already exists and is over; pass resume='allow' to continue it",
            404: f"run {self._id} does not exist on the server (resume='must')",
            422: "the server rejected the run description",
            426: "the server does not speak protocol version 1; upgrade airon or the server",
        }.get(result.status or 0, result.message)
        if result.status and 300 <= result.status < 400:
            reason = f"the server redirected the request ({result.message}); check AIRON_URL"
        return f"cannot log this run: {reason}. Nothing more is sent; data is saved to disk."

    def _give_up(self, kind: str, message: str) -> None:
        self._dead = True
        self._spooling = True
        _log.warn_once(kind, message)

    def _start_spooling(self, result: Result) -> None:
        if self._dead or self._abandoned:
            return
        self._spooling = True
        self._had_outage = True
        self._failed_probes = 1
        self._next_probe = time.monotonic() + self._backoff(1, result.retry_after)
        _log.warn_once(
            "network",
            f"cannot reach the Airon server ({result.message or 'unknown error'}). Training is not "
            "affected; data is saved to disk and sent when the connection returns.",
        )

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.t.retry_max_delay, self.t.retry_min_delay * 2.0 ** (attempt - 1))
        delay *= random.uniform(0.5, 1.0)
        if retry_after is not None:
            delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
        return delay

    def _sleep(self, seconds: float) -> None:
        """Sleep, but never past the shutdown deadline once close() has been called."""
        end = time.monotonic() + seconds
        while not self._abandoned:
            limit = min(end, self._deadline) if self._closing else end
            remaining = limit - time.monotonic()
            if remaining <= 0:
                return
            self._interrupt.wait(remaining)  # close() wakes this so the limit is re-read
            self._interrupt.clear()

    # ---- run creation, spool upload, heartbeat, finish ---------------------------------------

    def _ensure_created(self) -> Result | None:
        client = self._client
        if self._created or client is None or self._payload is None:
            return None
        result = client.create_run(self._payload)
        if not result.ok:
            return result
        self._created = True
        data = result.data
        try:
            self._seq = max(self._seq, int(data.get("last_seq") or 0))
        except (TypeError, ValueError):
            pass
        for warning in data.get("warnings") or []:
            _log.warn_once(f"server-{warning}", f"server: {warning}")
        self._on_created(data)
        return None

    def _replay(self, force: bool) -> None:
        """While disconnected, try to upload the spool, oldest segment first."""
        client = self._client
        if client is None or self._dead or not self._spooling:
            return
        if not force and time.monotonic() < self._next_probe:
            return
        while not self._abandoned:
            segment = self._spool.next_segment()
            events = self._spool.read_segment(segment) if segment else []
            if segment is not None and not events:
                self._spool.remove_segment(segment)
                continue
            call: Callable[[], Result] = (
                partial(send_split, client, self._id, events) if events else partial(Result, OK)
            )
            verdict, result = self._attempt(call, "events", attempts=1)
            if verdict == "spool":
                self._failed_probes += 1
                delay = self._backoff(self._failed_probes, result.retry_after)
                self._next_probe = time.monotonic() + delay
                return
            if segment is None:
                self._spooling = False
                self._failed_probes = 0
                if self._had_outage:
                    _log.say("connection restored; saved events were uploaded")
                    self._had_outage = False
                return
            if verdict == "ok":
                self._record_ack(result, len(events))
            else:
                self.stats.dropped += len(events)
            self._spool.remove_segment(segment)

    def _heartbeat(self) -> None:
        if self._spooling or self._closing:
            return
        if time.monotonic() - self._last_sent >= self.t.heartbeat_interval:
            self._last_sent = time.monotonic()
            self.enqueue({"t": time.time()})

    def _finalize(self) -> bool:
        finish = self._finish_req or {}
        client = self._client
        if client is not None and self._spooling and not self._dead:
            self._replay(force=True)  # one last try; the rest stays on disk for `airon sync`
        if client is None or self._spooling or self._dead:
            self._spool.write_finish(finish)
        else:
            verdict, _ = self._attempt(
                lambda: client.finish_run(self._id, finish), "finish request"
            )
            if verdict == "spool":
                self._spool.write_finish(finish)
        self._finish_req = None
        self._spool.remove_if_empty()
        return True


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return ""
