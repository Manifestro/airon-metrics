"""Process-wide hooks: unhandled exceptions, normal exit, SIGTERM and fork.

Existing handlers are always chained, never replaced: ours run around them.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ._run import Run

_lock = threading.Lock()
_runs: list[Run] = []
_installed = False
_signal_installed = False
_previous_excepthook: Callable[..., Any] | None = None
_previous_sigterm: Any = None


def register(run: Run) -> None:
    with _lock:
        _runs.append(run)
    _install()


def unregister(run: Run) -> None:
    with _lock:
        if run in _runs:
            _runs.remove(run)


def active() -> list[Run]:
    with _lock:
        return list(_runs)


def _install() -> None:
    global _installed, _previous_excepthook
    with _lock:
        first = not _installed
        _installed = True
    if first:
        _previous_excepthook = sys.excepthook
        sys.excepthook = _excepthook
        atexit.register(_at_exit)
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_after_fork)
    _install_sigterm()


def _excepthook(
    exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
) -> None:
    previous = _previous_excepthook or sys.__excepthook__
    try:
        previous(exc_type, exc, tb)  # print the traceback before we spend time flushing
    finally:
        for run in active():
            run._finish_from_exception(exc_type, exc, tb)


def _at_exit() -> None:
    # If another hook replaced ours, Python still records the unhandled exception here.
    last = getattr(sys, "last_value", None)
    for run in active():
        if last is not None:
            run._finish_from_exception(type(last), last, last.__traceback__)
        else:
            run._finish_at_exit()


def _after_fork() -> None:
    # A forked child shares the run object but not its threads; it must not report for the parent.
    for run in active():
        run._disable_in_child()
    with _lock:
        _runs.clear()


def _install_sigterm() -> None:
    global _signal_installed, _previous_sigterm
    if _signal_installed or threading.current_thread() is not threading.main_thread():
        return
    try:
        previous = signal.getsignal(signal.SIGTERM)
        if previous is None or previous == signal.SIG_IGN:
            return  # installed outside Python, or deliberately ignored: leave it alone
        _previous_sigterm = previous
        signal.signal(signal.SIGTERM, _on_sigterm)
        _signal_installed = True
    except (ValueError, OSError):
        return


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    previous = _previous_sigterm
    final = not callable(previous)
    try:
        if callable(previous):
            previous(signum, frame)  # the program's own handler goes first
    except BaseException:
        final = True  # the handler is ending the program (SystemExit and the like)
        raise
    finally:
        for run in active():
            run._finish_from_signal(final)
    if not callable(previous):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
