"""A job's inputs and outputs on a rented machine.

Inputs (`inputs: {data: ..., models: [...]}` in airon.yaml) are downloaded before the work
starts, from links Airon gives page by page: the script finds them in `$AIRON_DATA` and
`$AIRON_MODELS` and never deals with S3. On a split job each shard is given only its own part of
the data.

Outputs are whatever the `run` command leaves in `$AIRON_OUTPUT`. Every file is uploaded, keeping
its path inside the folder, into the store of the machine's region; Airon turns them into the
job's result (a dataset in the catalogue, when airon.yaml asks for one).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ._checkpoints import _put
from ._transport import Client

WORKERS = 8
RETRIES = 5
CHUNK = 8 * 1024 * 1024
Say = Callable[[str], None]


class TransferFailed(Exception):
    """An input could not be downloaded or an output uploaded, after retries."""


def _safe(base: Path, relative: str) -> Path:
    """`relative` under `base`, refusing anything that would leave it."""
    target = (base / relative).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise TransferFailed(f"refused a path outside the folder: {relative!r}")
    return target


def _download(url: str, target: Path, size: int | None) -> int:
    """One file, streamed to disk and kept when it is already there whole. Returns its size."""
    if size is not None and target.exists() and target.stat().st_size == size:
        return size  # a retry on the same machine: already downloaded
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
                while chunk := response.read(CHUNK):
                    out.write(chunk)
            got = partial.stat().st_size
            if size is not None and got != size:
                raise OSError(f"got {got} bytes of {size}")
            partial.replace(target)
            return got
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(min(30, 2**attempt))
    raise TransferFailed(f"could not download {target.name}: {last}")


def fetch_inputs(client: Client, folder: Path, say: Say) -> dict[str, Any]:
    """Download every input file Airon lists into `folder` (data/... and models/...)."""
    folder.mkdir(parents=True, exist_ok=True)
    count = total = 0
    cursor: str | None = None
    started = time.monotonic()
    last_said = 0.0
    while True:
        page = client.request("POST", "/v1/job/inputs", {"cursor": cursor}, timeout=120)
        if not page.ok:
            raise TransferFailed(f"Airon did not list the inputs: {page.message}")
        data = page.data or {}
        files = data.get("files") or []
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [
                pool.submit(
                    _download,
                    str(f["url"]),
                    _safe(folder, str(f["path"])),
                    int(f["size"]) if f.get("size") is not None else None,
                )
                for f in files
            ]
            for future in as_completed(futures):
                total += future.result()
                count += 1
                if time.monotonic() - last_said > 10:
                    last_said = time.monotonic()
                    # Read by Airon as the stage's progress bar.
                    say(f"::airon progress name=inputs done={count} unit=files")
        cursor = data.get("next")
        if not cursor:
            break
    say(f"::airon progress name=inputs done={count} total={count} unit=files")
    say(f"airon: inputs: {count} files, {total / 1e9:.1f} GB in {time.monotonic() - started:.0f} s")
    return {"files": count, "bytes": total}


def _upload_one(client: Client, folder: Path, file: Path) -> int:
    relative = file.relative_to(folder).as_posix()
    size = file.stat().st_size
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            started = client.request(
                "POST",
                "/v1/job/checkpoint/start",
                {"size": size, "step": 0, "filename": relative, "kind": "output"},
                timeout=120,
            )
            if not started.ok:
                raise TransferFailed(f"Airon refused the upload of {relative}: {started.message}")
            plan = started.data or {}
            part_size = int(plan.get("part_size") or size or 1)
            etags = []
            with file.open("rb") as handle:
                for part in plan["parts"]:
                    handle.seek((int(part["number"]) - 1) * part_size)
                    body = handle.read(part_size)
                    etags.append({"number": part["number"], "etag": _put(part["url"], body)})
            finished = client.request(
                "POST",
                "/v1/job/checkpoint/finish",
                {
                    "key": plan["key"],
                    "upload_id": plan.get("upload_id"),
                    "parts": etags,
                    "step": 0,
                    "label": "output",
                    "metrics": {},
                },
                timeout=300,
            )
            if not finished.ok:
                raise TransferFailed(f"Airon did not record {relative}: {finished.message}")
            return size
        except TransferFailed as exc:
            last = exc
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(min(30, 2**attempt))
    raise TransferFailed(f"could not upload {relative}: {last}")


def upload_outputs(client: Client, folder: Path, say: Say) -> dict[str, Any]:
    """Upload every file under `folder`, several at a time. Returns how many and how big."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
    if not files:
        return {"files": 0, "bytes": 0}
    total = 0
    done = 0
    lock = threading.Lock()
    last_said = [0.0]

    def one(file: Path) -> None:
        nonlocal total, done
        size = _upload_one(client, folder, file)
        with lock:
            total += size
            done += 1
            if time.monotonic() - last_said[0] > 10:
                last_said[0] = time.monotonic()
                say(f"::airon progress name=outputs done={done} total={len(files)} unit=files")

    with ThreadPoolExecutor(WORKERS) as pool:
        for future in as_completed([pool.submit(one, f) for f in files]):
            future.result()
    say(f"::airon progress name=outputs done={len(files)} total={len(files)} unit=files")
    say(f"airon: outputs: {len(files)} files, {total / 1e9:.2f} GB uploaded")
    return {"files": len(files), "bytes": total}


def setup_command(root: Path) -> str | None:
    """How to install the project's dependencies, from the lock file it has; None without one.
    The environment is `.venv` in the checkout, which every later stage uses."""
    if (root / "uv.lock").exists():
        # uv itself lives outside the project, so `uv sync` cannot remove it.
        tool = "$HOME/.airon-uv"
        return (
            f'[ -x "{tool}/bin/uv" ] || (python3 -m venv "{tool}" && '
            f'"{tool}/bin/pip" install --quiet uv) && "{tool}/bin/uv" sync --frozen'
        )
    if (root / "requirements.txt").exists():
        return (
            "python3 -m venv .venv && .venv/bin/pip install --quiet --upgrade pip && "
            ".venv/bin/pip install --quiet -r requirements.txt"
        )
    if (root / "pyproject.toml").exists():
        return (
            "python3 -m venv .venv && .venv/bin/pip install --quiet --upgrade pip && "
            ".venv/bin/pip install --quiet -e ."
        )
    return None


def venv_env(root: Path) -> dict[str, str]:
    """PATH and VIRTUAL_ENV for the checkout's `.venv`, when it has one."""
    venv = root / ".venv"
    if not (venv / "bin").is_dir():
        return {}
    return {
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
