"""The offline spool: events that could not be delivered, kept as JSON lines on disk.

Layout of ``<spool root>/<run id>/``::

    meta.json                  the create request, so the run can be recreated by `airon sync`
    events-000000000123.jsonl  segments of at most 500 events, named after their first seq
    finish.json                the finish request, when it could not be delivered

Every method swallows disk errors and reports them through its return value.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

SEGMENT_MAX_EVENTS = 500
SEGMENT_MAX_BYTES = 700_000
_SEGMENT = re.compile(r"^events-(\d+)\.jsonl$")


def directory_name(run_id: str) -> str:
    """A safe directory name for a run id, whatever the user passed."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", run_id).lstrip(".")[:128]
    return safe or "run"


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


class Spool:
    def __init__(self, directory: Path):
        self.directory = directory
        self.last_error = ""
        self._lock = threading.Lock()
        self._meta_written = False
        self._segment: Path | None = None
        self._segment_events = 0
        self._segment_bytes = 0

    def _fail(self, exc: BaseException) -> bool:
        self.last_error = f"{type(exc).__name__}: {exc}"
        return False

    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def read_meta(self) -> dict[str, Any] | None:
        return self._read_json(self.directory / "meta.json")

    def write_meta(self, meta: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._ensure_dir()
                self._write_json(self.directory / "meta.json", meta)
            except (OSError, ValueError, TypeError) as exc:
                return self._fail(exc)
            self._meta_written = True
            return True

    def max_seq(self) -> int:
        """Highest seq already on disk, so a restarted process continues the numbering."""
        last = 0
        for path in self.segments()[-1:]:
            for event in self.read_segment(path):
                last = max(last, int(event.get("seq") or 0))
        return last

    def append(self, events: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> bool:
        if not events:
            return True
        with self._lock:
            try:
                self._ensure_dir()
                if meta is not None and not self._meta_written:
                    self._write_json(self.directory / "meta.json", meta)
                    self._meta_written = True
                chunk: list[str] = []
                first_seq = int(events[0].get("seq") or 0)
                size = 0
                for event in events:
                    line = dumps(event)
                    if chunk and (
                        len(chunk) >= SEGMENT_MAX_EVENTS or size + len(line) > SEGMENT_MAX_BYTES
                    ):
                        self._write_chunk(chunk, size, first_seq)
                        chunk, size, first_seq = [], 0, int(event.get("seq") or 0)
                    chunk.append(line)
                    size += len(line) + 1
                self._write_chunk(chunk, size, first_seq)
            except (OSError, ValueError, TypeError) as exc:
                return self._fail(exc)
            return True

    def _write_chunk(self, lines: list[str], size: int, first_seq: int) -> None:
        if (
            self._segment is None
            or self._segment_events + len(lines) > SEGMENT_MAX_EVENTS
            or self._segment_bytes + size > SEGMENT_MAX_BYTES
        ):
            self._segment = self.directory / f"events-{first_seq:012d}.jsonl"
            self._segment_events = 0
            self._segment_bytes = 0
        with open(self._segment, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._segment_events += len(lines)
        self._segment_bytes += size

    def segments(self) -> list[Path]:
        try:
            names = sorted(n for n in os.listdir(self.directory) if _SEGMENT.match(n))
        except OSError:
            return []
        return [self.directory / n for n in names]

    def next_segment(self) -> Path | None:
        """The oldest segment. Appends move on to a new file, so it is safe to upload."""
        with self._lock:
            segments = self.segments()
            if not segments:
                return None
            if segments[0] == self._segment:
                self._segment = None
            return segments[0]

    def read_segment(self, path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # a torn last line after a kill
                    if isinstance(event, dict):
                        events.append(event)
        except OSError as exc:
            self._fail(exc)
        return events

    def remove_segment(self, path: Path) -> None:
        with self._lock:
            try:
                path.unlink()
            except OSError:
                pass

    def write_finish(self, payload: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._ensure_dir()
                self._write_json(self.directory / "finish.json", payload)
            except (OSError, ValueError, TypeError) as exc:
                return self._fail(exc)
            return True

    def read_finish(self) -> dict[str, Any] | None:
        return self._read_json(self.directory / "finish.json")

    def has_data(self) -> bool:
        return bool(self.segments()) or (self.directory / "finish.json").exists()

    def remove(self) -> None:
        """Delete the whole run directory (after a complete upload)."""
        with self._lock:
            shutil.rmtree(self.directory, ignore_errors=True)

    def remove_if_empty(self) -> None:
        """Clean up the metadata of a run that delivered everything."""
        if not self.has_data():
            self.remove()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(dumps(value), encoding="utf-8")
        os.replace(tmp, path)


def list_spools(root: Path) -> list[Spool]:
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [Spool(root / n) for n in names if (root / n / "meta.json").is_file()]
