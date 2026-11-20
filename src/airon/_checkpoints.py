"""Checkpoints kept by Airon: uploaded in the background, downloaded on demand.

Inside a job, `run.checkpoint(path)` copies the file (or packs the folder) right away, so the
training loop may overwrite or delete it at once, and a background thread uploads the copy to
Airon's storage through presigned URLs: the machine holds no storage credentials. `resume()` and
`fetch()` download any checkpoint Airon keeps, from any machine. Standard library only.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import queue
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import _config, _log
from ._transport import Client

PART_RETRIES = 4
CHUNK = 8 * 1024 * 1024
TAR_SUFFIX = ".airon.tar"


def _cache_root() -> Path:
    base = _config.env("AIRON_DIR")
    return Path(base) if base else Path.home() / ".cache" / "airon"


def job_client() -> Client | None:
    """The job key's client, inside a job; None elsewhere."""
    url, key = _config.env("AIRON_URL"), _config.env("AIRON_API_KEY")
    if not (url and key and _config.env("AIRON_JOB")):
        return None
    return Client(url, key, timeout=60.0)


def _plain(data: dict[str, Any] | None, message: str) -> str:
    return str((data or {}).get("error") or message)


class Uploader:
    """Uploads checkpoints one after another on a background thread. `submit` copies first, so
    the caller may change the file as soon as it returns; `wait` blocks until all are stored."""

    def __init__(
        self,
        client: Client,
        stage: str | None = None,
        fallback: Callable[[str, int | None, str | None, dict[str, Any]], None] | None = None,
    ):
        self._client = client
        self._stage = stage
        self._fallback = fallback
        # Set when the server says it keeps no checkpoints: no point asking again.
        self.disabled = False
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="airon-checkpoints", daemon=True)
        self._thread.start()
        self.stored: list[dict[str, Any]] = []
        self.failed: list[str] = []

    def submit(
        self,
        path: str,
        step: int | None,
        label: str | None,
        metrics: dict[str, Any],
        kind: str = "checkpoint",
    ) -> bool:
        if self.disabled:
            return False
        source = Path(path)
        if not source.exists():
            _log.warn_once(f"ckpt-missing-{path}", f"checkpoint {path} does not exist; not stored")
            return False
        staging = Path(tempfile.mkdtemp(prefix="upload-", dir=_staging_dir()))
        try:
            if source.is_dir():
                copy = staging / (source.name + TAR_SUFFIX)
                with tarfile.open(copy, "w") as tar:
                    tar.add(source, arcname=source.name)
            else:
                copy = staging / source.name
                shutil.copy2(source, copy)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            _log.warn_once("ckpt-copy", f"could not copy checkpoint {path} for upload: {exc}")
            return False
        if label is None and self._stage == "dry":
            label = "dry"
        self._queue.put(
            {
                "file": copy,
                "staging": staging,
                "source": str(path),
                "step": step,
                "label": "output" if kind == "output" else label,
                "metrics": metrics,
                "kind": kind,
            }
        )
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every submitted checkpoint is stored (or given up). True when done."""
        done = threading.Event()

        def drained() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=drained, daemon=True).start()
        return done.wait(timeout)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._upload(item)
            except Exception as exc:  # never let one checkpoint stop the next
                self.failed.append(str(item and item.get("file")))
                _log.warn_once("ckpt-upload", f"checkpoint not stored in Airon: {exc}")
                if item is not None and self._fallback is not None:
                    try:  # at least say where it is on this machine
                        self._fallback(item["source"], item["step"], item["label"], item["metrics"])
                    except Exception:
                        pass
            finally:
                if item is not None:
                    shutil.rmtree(item["staging"], ignore_errors=True)
                self._queue.task_done()

    def _upload(self, item: dict[str, Any]) -> None:
        file: Path = item["file"]
        size = file.stat().st_size
        started = self._client.request(
            "POST",
            "/v1/job/checkpoint/start",
            {"size": size, "step": item["step"], "filename": file.name, "kind": item.get("kind")},
            timeout=120,
        )
        if not started.ok:
            if started.status == 503:
                self.disabled = True
            raise RuntimeError(_plain(started.data, started.message))
        plan = started.data or {}
        part_size = int(plan.get("part_size") or size or 1)
        etags = []
        with file.open("rb") as handle:
            for part in plan["parts"]:
                handle.seek((int(part["number"]) - 1) * part_size)
                body = handle.read(part_size)
                etags.append({"number": part["number"], "etag": _put(part["url"], body)})
        finished = self._client.request(
            "POST",
            "/v1/job/checkpoint/finish",
            {
                "key": plan["key"],
                "upload_id": plan.get("upload_id"),
                "parts": etags,
                "step": item["step"],
                "label": item["label"],
                "metrics": item["metrics"],
            },
            timeout=300,
        )
        if not finished.ok:
            raise RuntimeError(_plain(finished.data, finished.message))
        self.stored.append(finished.data or {})
        _log.say(f"checkpoint stored: {(finished.data or {}).get('uri')}")


def _staging_dir() -> str:
    folder = _cache_root() / "staging"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _put(url: str, body: bytes) -> str:
    """PUT one part to its presigned URL, retrying; returns the ETag S3 gave it."""
    last: Exception | None = None
    for attempt in range(PART_RETRIES):
        request = urllib.request.Request(url, data=body, method="PUT")
        request.add_header("Content-Length", str(len(body)))
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return str(response.headers.get("ETag", "")).strip()
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"could not upload a part: {last}")


# --- reading -------------------------------------------------------------------------------------


def fetch(uri: str, say: Callable[[str], None] | None = None) -> str:
    """Download a checkpoint Airon keeps (an s3:// address it gave) and return its local path; a
    packed folder comes back unpacked. Anything else is returned as it is. Downloads are cached,
    so asking twice costs nothing. Works inside a job (job key) and anywhere `airon login` was
    done (a person's or agent's key)."""
    if not uri.startswith("s3://"):
        return uri
    target = _cache_root() / "checkpoints" / hashlib.sha256(uri.encode()).hexdigest()[:24]
    name = uri.rstrip("/").rsplit("/", 1)[-1]
    unpacked = name.endswith(TAR_SUFFIX)
    final = target / (name[: -len(TAR_SUFFIX)] if unpacked else name)
    if (target / ".complete").exists() and final.exists():
        return str(final)
    connection = _config.resolve_connection()
    if not connection.complete:
        raise RuntimeError("no Airon server configured (AIRON_URL and AIRON_API_KEY)")
    assert connection.url and connection.api_key
    client = Client(connection.url, connection.api_key, timeout=60.0)
    path = "/v1/job/checkpoint/download" if _config.env("AIRON_JOB") else "/v1/checkpoints/download"
    answer = client.request("POST", path, {"uri": uri}, timeout=60)
    if not answer.ok:
        raise RuntimeError(f"Airon would not give {uri}: {_plain(answer.data, answer.message)}")
    url = (answer.data or {})["url"]
    target.mkdir(parents=True, exist_ok=True)
    partial = target / (name + ".part")
    (say or _log.say)(f"downloading checkpoint {uri}")
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
    if unpacked:
        with tarfile.open(partial) as tar:
            if hasattr(tarfile, "data_filter"):  # refuses paths that would escape the folder
                tar.extractall(target, filter="data")
            else:  # Python without extraction filters: check the paths ourselves
                for member in tar.getmembers():
                    if member.name.startswith(("/", "..")) or "/../" in member.name:
                        raise RuntimeError(f"unsafe path in checkpoint: {member.name}")
                tar.extractall(target)
        partial.unlink()
    else:
        os.replace(partial, final)
    (target / ".complete").write_text(uri)
    return str(final)


# --- shards and their results ------------------------------------------------------------------

_outputs: Uploader | None = None
_outputs_lock = threading.Lock()


def shard_info() -> tuple[int, int]:
    """(this machine's shard, how many shards), from the job; (0, 1) when the job is not split."""
    try:
        count = max(1, int(_config.env("AIRON_SHARDS") or 1))
        index = int(_config.env("AIRON_SHARD") or 0)
    except ValueError:
        return 0, 1
    return (index, count) if 0 <= index < count else (0, 1)


def shard(items: Any) -> list[Any]:
    """This machine's part of `items`: every n-th item starting at its shard number. Give the
    items in the same order on every machine (sort file names, for example)."""
    index, count = shard_info()
    return list(items)[index::count]


def output(path: str) -> bool:
    """Store a result file or folder of this job in Airon's store, under the job's outputs, where
    the merge stage and anyone with a key can read it; never deleted with old checkpoints. The
    file is copied at once and uploaded in the background; the process waits for it before it
    ends. Outside a job it does nothing and says so."""
    global _outputs
    client = job_client()
    if client is None:
        _log.warn_once("output-no-job", "airon.output() stores results only inside a job; ignored")
        return False
    with _outputs_lock:
        if _outputs is None:
            _outputs = Uploader(client, _config.env("AIRON_STAGE"))
            atexit.register(_wait_for_outputs)
        return _outputs.submit(path, None, None, {}, kind="output")


def _wait_for_outputs() -> None:
    if _outputs is not None:
        _outputs.wait(float(_config.env("AIRON_CHECKPOINT_WAIT") or 3600))


def outputs() -> list[dict[str, Any]]:
    """The results every shard of this job stored with `output()`: [{uri, name, shard, bytes}].
    Download one with `fetch(uri)`. Meant for the merge stage."""
    _wait_for_outputs()
    client = job_client()
    if client is None:
        raise RuntimeError("airon.outputs() works inside a job")
    answer = client.request("GET", "/v1/job/outputs", timeout=60)
    if not answer.ok:
        raise RuntimeError(
            f"Airon would not list the outputs: {_plain(answer.data, answer.message)}"
        )
    return list((answer.data or {}).get("outputs") or [])


def resume() -> str | None:
    """The checkpoint to continue from when Airon asks for one (AIRON_RESUME), downloaded and
    ready to load; None when there is none. A download that fails is reported and gives None, so
    training starts over rather than not at all; the dry run catches a resume that does not work."""
    uri = _config.env("AIRON_RESUME")
    if not uri:
        return None
    try:
        return fetch(uri)
    except Exception as exc:
        _log.warn_once("resume", f"could not fetch the checkpoint to resume from ({exc})")
        return None
