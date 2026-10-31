"""Turning whatever the user logs into JSON the protocol accepts."""

from __future__ import annotations

import math
import numbers
import re
from pathlib import PurePath
from typing import Any, Callable

MAX_NAME_LENGTH = 200
_MAX_DEPTH = 8
_SECRET_KEY = re.compile(
    r"(?i)(token|secret|passw(or)?d|api[_-]?key|credential|authorization|private[_-]?key)"
)


def number(value: Any) -> int | float | str | None:
    """Return a JSON-safe number for ``value``, or None if it is not numeric.

    Accepts Python numbers and anything with a scalar ``.item()`` (numpy, torch) without
    importing those libraries. Non-finite floats become the strings the protocol defines.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _non_finite(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            inner = item()
        except Exception:
            return None
        if isinstance(inner, (bool, int, float)):
            return number(inner)
        return None
    if isinstance(value, numbers.Real):
        try:
            return number(float(value))
        except Exception:
            return None
    return None


def _non_finite(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def metrics(raw: Any, warn: Callable[[str, str], None]) -> dict[str, int | float | str]:
    """Clean a metrics dict. Bad entries are dropped with one warning per kind."""
    out: dict[str, int | float | str] = {}
    if not isinstance(raw, dict):
        warn("metrics-type", "log() expects a dict of metric names to numbers; ignoring this call")
        return out
    _collect(raw, "", out, warn, 0)
    return out


def _collect(
    raw: dict[Any, Any],
    prefix: str,
    out: dict[str, int | float | str],
    warn: Callable[[str, str], None],
    depth: int,
) -> None:
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth < 4:
            _collect(value, name + "/", out, warn, depth + 1)
            continue
        if not name or len(name) > MAX_NAME_LENGTH:
            warn(
                "metric-name",
                f"metric name is empty or longer than {MAX_NAME_LENGTH} characters; dropped",
            )
            continue
        clean = number(value)
        if clean is None:
            warn(
                "metric-value",
                f"metric {name!r} is not a number ({type(value).__name__}); dropped. "
                "Further non-numeric values are dropped silently.",
            )
            continue
        out[name] = clean


def jsonable(value: Any, depth: int = 0) -> Any:
    """Convert an arbitrary object to something ``json.dumps`` accepts (never raises)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return number(value)
    if depth >= _MAX_DEPTH:
        return repr(value)[:200]
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, depth + 1) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    for method in ("item", "tolist"):
        convert = getattr(value, method, None)
        if callable(convert):
            try:
                result = convert()
            except Exception:
                continue
            if not callable(getattr(result, method, None)):
                return jsonable(result, depth + 1)
    return repr(value)[:200]


def scrub(value: Any, depth: int = 0) -> Any:
    """``jsonable`` plus masking of values whose key looks like a secret."""
    clean = jsonable(value, depth)
    return _mask(clean)


def _mask(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "***" if _SECRET_KEY.search(k) and v not in (None, "") else _mask(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value
