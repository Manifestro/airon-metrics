"""The Run object and ``airon.init``."""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from collections.abc import Iterable, Iterator, MutableMapping
from types import TracebackType
from typing import Any

from . import _config, _env, _hooks, _log, _values
from ._console import ConsoleCapture
from ._names import friendly_name
from ._sender import Sender, Tuning
from ._spool import Spool, directory_name
from ._system import SystemSampler
from ._transport import Client

_RATE_SMOOTHING = 0.3  # weight of the newest steps/sec reading
_TRACEBACK_LIMIT = 20_000


class _Section(MutableMapping):  # type: ignore[type-arg]
    """``run.config`` and ``run.summary``: a dict whose assignments are sent to the server."""

    def __init__(self, initial: dict[str, Any], sink: Any):
        object.__setattr__(self, "_data", dict(initial))
        object.__setattr__(self, "_sink", sink)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def update(self, *args: Any, **kwargs: Any) -> None:
        changes = _values.scrub(dict(*args, **kwargs))
        self._data.update(changes)
        self._sink(changes)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __repr__(self) -> str:
        return repr(self._data)


class Run:
    """One tracked training run. Create it with ``airon.init``.

    Every method is safe to call from the training loop: none of them blocks on the network or
    raises because of it.
    """

    def __init__(
        self,
        run_id: str,
        name: str,
        project: str,
        mode: str,
        url: str | None,
        config: dict[str, Any],
        total_steps: int | None,
        total_epochs: float | None,
        spool: Spool,
    ):
        self.id = run_id
        self.name = name
        self.project = project
        self.mode = mode
        self.url = url
        self._spool = spool
        self._total_steps = total_steps
        self._total_epochs = total_epochs
        self.config = _Section(config, lambda change: self._event({"config": change}))
        self.summary = _Section({}, lambda change: self._event({"summary": change}))

        self._lock = threading.Lock()
        self._finish_guard = threading.Lock()
        self._finished = False
        self._child = False
        self._signalled = False
        self._created = threading.Event()
        self._next_step = 0
        self._last_step: int | None = None
        self._log_calls = 0
        self._epoch: float | None = None
        self._previous: tuple[int, float] | None = None
        self._steps_per_sec: float | None = None
        self._sender: Sender
        self._console: ConsoleCapture | None = None
        self._sampler: SystemSampler | None = None

    # ---- public API ----------------------------------------------------------------------

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        epoch: float | None = None,
        samples: int | None = None,
        tokens: int | None = None,
    ) -> None:
        """Record metrics.

        ``step`` defaults to a counter that starts at 0 and grows by one per call; passing a step
        moves the counter to ``step + 1``. ``samples`` and ``tokens`` are the amounts processed by
        one training step: with the measured steps per second they give samples and tokens per
        second.
        """
        if self._finished or self._child:
            return
        try:
            clean = _values.metrics(metrics, _log.warn_once)
            wall, tick = time.time(), time.monotonic()
            with self._lock:
                explicit = _as_int(step)
                step_value = self._next_step if explicit is None else explicit
                self._next_step = step_value + 1
                self._last_step = step_value
                self._log_calls += 1
                epoch_value = _as_float(epoch)
                if epoch_value is not None:
                    self._epoch = epoch_value
                progress = self._progress(step_value, tick, _as_float(samples), _as_float(tokens))
            event: dict[str, Any] = {"t": wall, "step": step_value}
            if epoch_value is not None:
                event["epoch"] = epoch_value
            if clean:
                event["metrics"] = clean
            event["progress"] = progress
            self._sender.enqueue(event)
        except Exception as exc:
            _log.warn_once("log-internal", f"log() failed ({type(exc).__name__}: {exc}); ignored")

    def progress(
        self,
        step: int | None = None,
        total_steps: int | None = None,
        epoch: float | None = None,
        total_epochs: float | None = None,
    ) -> None:
        """Report progress without metrics, for example at the start of an epoch."""
        if self._finished or self._child:
            return
        try:
            tick = time.monotonic()
            with self._lock:
                if total_steps is not None:
                    self._total_steps = _as_int(total_steps)
                if total_epochs is not None:
                    self._total_epochs = _as_float(total_epochs)
                if epoch is not None:
                    self._epoch = _as_float(epoch)
                explicit = _as_int(step)
                current = self._last_step if explicit is None else explicit
                if explicit is not None:
                    self._last_step = explicit
                progress = self._progress(current, tick, None, None)
            self._event({"progress": progress})
        except Exception as exc:
            _log.warn_once(
                "log-internal", f"progress() failed ({type(exc).__name__}: {exc}); ignored"
            )

    def checkpoint(
        self, uri: str, step: int | None = None, label: str | None = None, **metrics: Any
    ) -> None:
        """Record where a checkpoint lives. Nothing is uploaded: the URI is only a pointer."""
        if self._finished or self._child:
            return
        try:
            point: dict[str, Any] = {"uri": str(uri)}
            at = _as_int(step)
            if at is None:
                at = self._last_step
            if at is not None:
                point["step"] = at
            if label is not None:
                point["label"] = str(label)
            clean = _values.metrics(metrics, _log.warn_once)
            if clean:
                point["metrics"] = clean
            self._event({"checkpoint": point})
        except Exception as exc:
            _log.warn_once(
                "log-internal", f"checkpoint() failed ({type(exc).__name__}: {exc}); ignored"
            )

    def finish(self, status: str | None = None, exit_code: int | None = None) -> None:
        """End the run: send what is queued (at most ``shutdown_timeout`` seconds) and its status.

        ``status`` is "finished" (default), "failed" or "killed".
        """
        self._finish(status, exit_code, None)

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None or exc is None:
            self.finish()
        else:
            self._finish_from_exception(exc_type, exc, tb)

    # ---- internals ---------------------------------------------------------------------------

    def _event(self, body: dict[str, Any]) -> None:
        if self._finished or self._child:
            return
        try:
            body["t"] = time.time()
            self._sender.enqueue(body)
        except Exception as exc:
            _log.warn_once("log-internal", f"internal error ({type(exc).__name__}: {exc}); ignored")

    def _progress(
        self, step: int | None, tick: float, samples: float | None, tokens: float | None
    ) -> dict[str, Any]:
        """Progress fields for an event. The caller holds the lock."""
        if step is not None:
            before = self._previous
            if before is None or step < before[0]:
                self._previous = (step, tick)  # first log, or the step counter restarted
                if before is not None:
                    self._steps_per_sec = None
            elif step > before[0] and tick > before[1]:
                # Steps over seconds since the last advance, so logging every 100 steps still
                # gives the right rate. Smoothed because evaluation pauses make single readings jumpy.
                rate = (step - before[0]) / (tick - before[1])
                old = self._steps_per_sec
                self._steps_per_sec = (
                    rate if old is None else (_RATE_SMOOTHING * rate + (1 - _RATE_SMOOTHING) * old)
                )
                self._previous = (step, tick)
        speed = self._steps_per_sec
        fields: dict[str, Any] = {
            "step": step,
            "total_steps": self._total_steps,
            "epoch": self._epoch,
            "total_epochs": self._total_epochs,
            "steps_per_sec": speed,
            "samples_per_sec": speed * samples if speed and samples else None,
            "tokens_per_sec": speed * tokens if speed and tokens else None,
        }
        return {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in fields.items()
            if v is not None
        }

    def _on_created(self, data: dict[str, Any]) -> None:
        if isinstance(data.get("url"), str):
            self.url = data["url"]
        if isinstance(data.get("name"), str) and data["name"]:
            self.name = data["name"]
        last_step = _as_int(data.get("last_step"))
        if data.get("resumed") and last_step is not None:
            with self._lock:
                if self._log_calls == 0:
                    self._next_step = last_step + 1
        self._created.set()

    def _console_events(self) -> list[dict[str, Any]]:
        lines = self._console.drain() if self._console else []
        return [{"t": time.time(), "log": lines}] if lines else []

    def _finish(
        self, status: str | None, exit_code: int | None, error: dict[str, Any] | None
    ) -> None:
        if self._child or not self._finish_guard.acquire(blocking=False):
            return  # a second call, or a forked child that must not speak for its parent
        try:
            if status is None:
                status = "killed" if self._signalled else "finished"
            elif status not in ("finished", "failed", "killed"):
                _log.warn_once("status", f"unknown status {status!r}; using 'finished'")
                status = "finished"
            self._finished = True
            _hooks.unregister(self)
            if self._sampler:
                self._sampler.stop()
            if self._console:
                self._console.uninstall()
            for event in self._console_events():
                self._sender.enqueue(event)
            request: dict[str, Any] = {"status": status, "t": time.time()}
            if exit_code is not None:
                request["exit_code"] = exit_code
            if error:
                request["error"] = error
            if self.summary:
                request["summary"] = dict(self.summary)
            self._sender.close(request)
            _forget(self)
            self._report(status, error)
        except Exception as exc:
            _log.warn_once("finish", f"finishing the run failed ({type(exc).__name__}: {exc})")

    def _report(self, status: str, error: dict[str, Any] | None) -> None:
        what = status if not error else f"{status} ({error.get('type')})"
        if self.mode == "offline":
            _log.say(f"run {self.name} {what}; upload it with `airon sync`")
            return
        _log.say(f"run {self.name} {what} -> {self.url}")
        if self._spool.has_data():
            _log.say(
                f"some data is saved in {self._spool.directory}; upload it with `airon sync` "
                "when the server is reachable"
            )

    def _finish_from_exception(
        self,
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, SystemExit) and getattr(exc, "code", None) in (None, 0):
            self._finish("finished", 0, None)  # sys.exit() inside the block is a normal end
            return
        interrupted = issubclass(exc_type, KeyboardInterrupt)
        module = getattr(exc_type, "__module__", "")
        name = (
            exc_type.__qualname__
            if module in ("builtins", "")
            else f"{module}.{exc_type.__qualname__}"
        )
        trace = "".join(traceback.format_exception(exc_type, exc, tb))
        error = {"type": name, "message": str(exc)[:2000], "traceback": trace[-_TRACEBACK_LIMIT:]}
        code = getattr(exc, "code", None)
        exit_code = 130 if interrupted else code if isinstance(code, int) else 1
        self._finish("killed" if interrupted else "failed", exit_code, error)

    def _finish_at_exit(self) -> None:
        self._finish(None, None, None)

    def _finish_from_signal(self, final: bool) -> None:
        self._signalled = True
        if final:
            self._finish("killed", None, {"type": "SIGTERM", "message": "terminated by SIGTERM"})
        elif not self._finished:
            self._sender.flush(timeout=3.0)  # the program may be killed for good any moment now

    def _disable_in_child(self) -> None:
        self._child = True


class NoopRun(Run):
    """Same API as ``Run``; every call does nothing. Used when disabled and on non-zero ranks."""

    def __init__(self, run_id: str = "", name: str = "", project: str = "", mode: str = "disabled"):
        self.id = run_id
        self.name = name
        self.project = project
        self.mode = mode
        self.url = None
        self.config = _Section({}, lambda change: None)
        self.summary = _Section({}, lambda change: None)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        epoch: float | None = None,
        samples: int | None = None,
        tokens: int | None = None,
    ) -> None:
        return None

    def progress(
        self,
        step: int | None = None,
        total_steps: int | None = None,
        epoch: float | None = None,
        total_epochs: float | None = None,
    ) -> None:
        return None

    def checkpoint(
        self, uri: str, step: int | None = None, label: str | None = None, **metrics: Any
    ) -> None:
        return None

    def finish(self, status: str | None = None, exit_code: int | None = None) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _as_int(value: Any) -> int | None:
    clean = _values.number(value) if value is not None else None
    if isinstance(clean, float) and clean.is_integer():
        return int(clean)
    return clean if isinstance(clean, int) else None


def _as_float(value: Any) -> float | None:
    clean = _values.number(value) if value is not None else None
    return float(clean) if isinstance(clean, (int, float)) else None


# ---- init and the module-level shortcuts -----------------------------------------------------

_current: Run | None = None
_init_lock = threading.Lock()


def current() -> Run | None:
    return _current


def _forget(run: Run) -> None:
    global _current
    if _current is run:
        _current = None


def _is_rank_zero() -> bool:
    for variable in ("RANK", "LOCAL_RANK"):
        raw = _config.env(variable)
        try:
            if raw is not None and int(raw) != 0:
                return False
        except ValueError:
            continue
    return True


def _resolve_mode(mode: str | None) -> str:
    chosen = (mode or _config.env("AIRON_MODE") or "online").lower()
    if chosen not in _config.MODES:
        _log.warn_once(
            "mode", f"unknown mode {chosen!r} (use online, offline or disabled); using online"
        )
        return "online"
    return chosen


def _resolve_resume(resume: Any) -> str:
    if resume is True:
        return "allow"
    if resume is False or resume is None:
        return "never"
    if resume in _config.RESUME_MODES:
        return str(resume)
    _log.warn_once(
        "resume", f"unknown resume value {resume!r} (use never, allow or must); using never"
    )
    return "never"


def init(
    project: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    experiment: str | None = None,
    tags: Iterable[str] | None = None,
    notes: str | None = None,
    total_steps: int | None = None,
    total_epochs: float | None = None,
    resume: str = "never",
    run_id: str | None = None,
    system_metrics: bool = True,
    console: bool = True,
    mode: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    rank_zero_only: bool = True,
    *,
    flush_interval: float = 2.0,
    heartbeat_interval: float = 30.0,
    system_interval: float = 10.0,
    batch_size: int = 200,
    max_queue: int = 100_000,
    request_timeout: float = 10.0,
    retry_min_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    retry_attempts: int = 4,
    shutdown_timeout: float = 10.0,
    console_lines: int = 200,
    init_timeout: float = 5.0,
) -> Run:
    """Start tracking a run. Never raises: on any problem you get a run that does nothing."""
    try:
        return _init(locals())
    except Exception as exc:
        _log.warn_once(
            "init", f"could not start tracking ({type(exc).__name__}: {exc}); continuing without it"
        )
        return NoopRun()


def _init(args: dict[str, Any]) -> Run:
    global _current
    _log.reset()
    chosen_mode = _resolve_mode(args["mode"])
    if chosen_mode == "disabled":
        return NoopRun(mode="disabled")
    if args["rank_zero_only"] and not _is_rank_zero():
        return NoopRun(mode="disabled")

    with _init_lock:
        previous = _current
        if previous is not None:
            previous.finish()

        connection = _config.resolve_connection(args["url"], args["api_key"])
        if chosen_mode == "online" and not connection.complete:
            _log.warn_once("config", _missing_config_message(connection))
            chosen_mode = "offline"

        run_id = str(args["run_id"] or _config.env("AIRON_RUN_ID") or uuid.uuid4())
        name = str(args["name"])[:200] if args["name"] else friendly_name()
        project = str(args["project"] or _config.env("AIRON_PROJECT") or _default_project())[:200]
        resume = _resolve_resume(args["resume"])
        tuning = Tuning(
            flush_interval=max(0.01, args["flush_interval"]),
            heartbeat_interval=max(0.05, args["heartbeat_interval"]),
            batch_size=max(1, min(500, args["batch_size"])),
            max_queue=max(10, args["max_queue"]),
            retry_min_delay=max(0.001, args["retry_min_delay"]),
            retry_max_delay=max(0.001, args["retry_max_delay"]),
            retry_attempts=max(1, args["retry_attempts"]),
            shutdown_timeout=max(0.0, args["shutdown_timeout"]),
        )
        config = _values.scrub(dict(args["config"] or {}))
        total_steps = _as_int(args["total_steps"])
        total_epochs = _as_float(args["total_epochs"])

        payload: dict[str, Any] = {
            "run_id": run_id,
            "project": project,
            "name": name,
            "experiment": args["experiment"],
            "config": config,
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "tags": [str(t) for t in args["tags"]] if args["tags"] else None,
            "notes": args["notes"],
            "resume": resume,
            "started_at": time.time(),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        online = chosen_mode == "online"
        client = (
            Client(connection.url or "", connection.api_key or "", args["request_timeout"])
            if online
            else None
        )
        spool = Spool(_config.spool_root() / directory_name(run_id))
        link = f"{connection.url}/runs/{run_id}" if online else None
        run = Run(
            run_id, name, project, chosen_mode, link, config, total_steps, total_epochs, spool
        )

        libs = _env.imported_libs()
        run._sender = Sender(
            run_id,
            spool,
            client,
            payload,
            tuning,
            build_env=lambda: _env.capture(libs),
            on_created=run._on_created,
            extras=run._console_events,
        )
        if args["console"]:
            run._console = ConsoleCapture(args["console_lines"])
        if args["system_metrics"]:
            run._sampler = SystemSampler(
                lambda values: run._event({"system": values}), args["system_interval"]
            )

        run._sender.start()
        if run._console:
            run._console.install()
        if run._sampler:
            run._sampler.start()
        _hooks.register(run)
        _current = run

    _announce(run)
    if online and resume != "never":
        run._created.wait(
            args["init_timeout"]
        )  # resuming needs the server's last_seq and last_step
    return run


def _announce(run: Run) -> None:
    if run.mode == "online":
        _log.say(f"tracking run {run.name} -> {run.url}")
    else:
        _log.say(
            f"run {run.name} is saved in {run._spool.directory} (offline); upload it with `airon sync`"
        )


def _default_project() -> str:
    return os.path.basename(os.getcwd()) or "default"


def _missing_config_message(connection: _config.Connection) -> str:
    missing = []
    if connection.raw_url is None:
        missing.append("AIRON_URL")
    elif connection.url is None:
        missing.append(
            f"AIRON_URL (the value {connection.raw_url!r} is not a valid http(s) address)"
        )
    if not connection.api_key:
        missing.append("AIRON_API_KEY")
    return (
        f"no usable server settings ({', '.join(missing)}), so this run is only saved on disk. "
        "Set AIRON_URL and AIRON_API_KEY, or run `airon login`, then upload with `airon sync`."
    )


def log(
    metrics: dict[str, Any],
    step: int | None = None,
    epoch: float | None = None,
    samples: int | None = None,
    tokens: int | None = None,
) -> None:
    """``run.log`` on the current run."""
    run = _current
    if run is None:
        _log.warn_once(
            "no-run", "airon.log() was called with no active run (call airon.init() first); ignored"
        )
        return
    run.log(metrics, step=step, epoch=epoch, samples=samples, tokens=tokens)


def finish(status: str | None = None, exit_code: int | None = None) -> None:
    """``run.finish`` on the current run."""
    run = _current
    if run is not None:
        run.finish(status, exit_code)
