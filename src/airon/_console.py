"""Capture the tail of stdout and stderr without changing what the program prints."""

from __future__ import annotations

import re
import sys
import threading
from collections import deque
from typing import Any

LINE_LIMIT = 4000
_SPLIT = re.compile(r"(\r\n|\n|\r)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _Tee:
    """Forwards everything to the real stream and shows the text to the capture."""

    def __init__(self, original: Any, name: str, capture: ConsoleCapture):
        self._original = original
        self._name = name
        self._capture = capture

    def write(self, text: str) -> int:
        written = self._original.write(text)
        try:
            self._capture.feed(self._name, text)
        except Exception:
            pass  # capturing must never turn into an error in the program's print()
        return len(text) if written is None else int(written)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __iter__(self) -> Any:
        return iter(self._original)


class _Assembler:
    """Builds lines from a stream of text, treating a lone carriage return as "start over"."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.overwrite = False

    def feed(self, text: str) -> list[str]:
        done: list[str] = []
        for piece in _SPLIT.split(text):
            if piece in ("\n", "\r\n"):
                done.append("".join(self.parts))
                self.parts = []
                self.overwrite = False
            elif piece == "\r":
                self.overwrite = True
            elif piece:
                if self.overwrite:
                    self.parts = []
                    self.overwrite = False
                self.parts.append(piece)
        return done

    def rest(self) -> str:
        text = "".join(self.parts)
        self.parts = []
        return text


class ConsoleCapture:
    def __init__(self, max_lines: int = 200):
        self._lock = threading.RLock()
        self._lines: deque[tuple[str, str]] = deque(maxlen=max(1, max_lines))
        self._assemblers = {"stdout": _Assembler(), "stderr": _Assembler()}
        self._omitted = 0
        self._tees: dict[str, _Tee] = {}
        self._originals: dict[str, Any] = {}
        self._active = False

    def install(self) -> None:
        for name in ("stdout", "stderr"):
            original = getattr(sys, name)
            self._originals[name] = original
            tee = _Tee(original, name, self)
            self._tees[name] = tee
            setattr(sys, name, tee)
        self._active = True

    def uninstall(self) -> None:
        for name, tee in self._tees.items():
            if getattr(sys, name) is tee:
                setattr(sys, name, self._originals[name])
        with self._lock:
            self._active = False
            for name, assembler in self._assemblers.items():
                self._add(name, assembler.rest())

    def feed(self, stream: str, text: str) -> None:
        if not self._active:
            return
        with self._lock:
            for line in self._assemblers[stream].feed(text):
                self._add(stream, line)

    def _add(self, stream: str, line: str) -> None:
        line = _ANSI.sub("", line).rstrip()
        if not line or line.startswith("airon: "):  # the SDK's own messages are not training output
            return
        if len(self._lines) == self._lines.maxlen:
            self._omitted += 1
        self._lines.append((stream, line[:LINE_LIMIT]))

    def drain(self) -> list[dict[str, str]]:
        """Lines captured since the last call, newest 200 at most."""
        with self._lock:
            lines = [{"stream": s, "text": t} for s, t in self._lines]
            omitted, self._omitted = self._omitted, 0
            self._lines.clear()
        if omitted:
            lines.insert(
                0, {"stream": "stderr", "text": f"[airon: {omitted} earlier lines omitted]"}
            )
        return lines
