"""Terminal messages. Everything the SDK says goes through here so keys can be redacted."""

from __future__ import annotations

import re
import sys
import threading

_lock = threading.Lock()
_seen: set[str] = set()
_secrets: list[str] = []
_BEARER = re.compile(r"(?i)(bearer\s+)\S+")


def add_secret(secret: str | None) -> None:
    # Very short strings would also blank out ordinary words in messages.
    if secret and len(secret) >= 6 and secret not in _secrets:
        _secrets.append(secret)


def redact(text: str) -> str:
    for secret in _secrets:
        text = text.replace(secret, "***")
    return _BEARER.sub(r"\1***", text)


def reset() -> None:
    """Forget which warnings were printed, so a new run can warn again."""
    with _lock:
        _seen.clear()


def _write(message: str) -> None:
    try:
        print(redact(message), file=sys.stderr, flush=True)
    except Exception:
        pass  # a closed or broken stderr must never reach the training code


def warn_once(kind: str, message: str) -> None:
    """Print a warning, at most once per kind of problem."""
    with _lock:
        if kind in _seen:
            return
        _seen.add(kind)
    _write(f"airon: warning: {message}")


def say(message: str) -> None:
    """Print an informational line unless AIRON_QUIET is set."""
    from ._config import env_flag

    if not env_flag("AIRON_QUIET"):
        _write(f"airon: {message}")
