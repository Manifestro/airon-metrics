"""Jobs that do not train (`type: run`): inputs downloaded before the work, results uploaded from
$AIRON_OUTPUT after it, dependencies installed from the lock file, and a retry on a fresh machine
that prepares everything again."""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from airon import _io, _job
from airon._transport import Client


class Airon:
    """Job endpoints, input links and the store behind presigned URLs, in one fake."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, str, dict[str, Any]]] = []
        self.lines: list[dict[str, Any]] = []
        self.inputs: dict[str, bytes] = {}  # path under the inputs folder -> content
        self.page_size = 2
        self.stored: dict[str, bytes] = {}  # uploaded outputs, by relative path
        self.parts: dict[str, dict[int, bytes]] = {}
        self.downloads = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, raw: bytes, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802  (an input's presigned link)
                fake.downloads += 1
                self._send(200, fake.inputs[self.path.removeprefix("/get/")])

            def do_PUT(self) -> None:  # noqa: N802  (an output's presigned part)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                key, _, number = self.path.removeprefix("/put/").partition("?part=")
                fake.parts.setdefault(key, {})[int(number)] = body
                self._send(200, b"", {"ETag": f'"e{number}"'})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                answer: dict[str, Any] = {}
                if self.path == "/v1/job/stage":
                    fake.stages.append((body["stage"], body["state"], body))
                elif self.path == "/v1/job/logs":
                    fake.lines.extend(body["lines"])
                elif self.path == "/v1/job/inputs":
                    paths = sorted(fake.inputs)
                    start = int(body.get("cursor") or 0)
                    page = paths[start : start + fake.page_size]
                    answer = {
                        "files": [
                            {"path": p, "url": f"{fake.url}/get/{p}", "size": len(fake.inputs[p])}
                            for p in page
                        ],
                        "next": str(start + fake.page_size)
                        if start + fake.page_size < len(paths)
                        else None,
                    }
                elif self.path == "/v1/job/checkpoint/start":
                    key = body["filename"]
                    answer = {
                        "key": key,
                        "part_size": 1024,
                        "parts": [
                            {"number": n, "url": f"{fake.url}/put/{key}?part={n}"}
                            for n in range(1, max(1, -(-body["size"] // 1024)) + 1)
                        ],
                    }
                elif self.path == "/v1/job/checkpoint/finish":
                    parts = fake.parts.pop(body["key"], {})
                    fake.stored[body["key"]] = b"".join(parts[n] for n in sorted(parts))
                    answer = {"uri": f"s3://store/{body['key']}"}
                elif self.path == "/v1/job/check":
                    answer = {"ok": True, "problems": [], "warnings": []}
                self._send(200, json.dumps(answer).encode(), {"Content-Type": "application/json"})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def states(self) -> list[tuple[str, str]]:
        return [(stage, state) for stage, state, _ in self.stages]


@pytest.fixture
def airon() -> Any:
    fake = Airon()
    yield fake
    fake.http.shutdown()


def make(fake: Airon, root: Path, stages: dict[str, str], **spec: Any) -> _job.Runner:
    job = {
        "job": "XP-01/1",
        "token": "job-key",
        "run_id": "00000000-0000-0000-0000-000000000001",
        "project": "lab",
        "experiment": "XP-01",
        "spec": {"stages": stages, **spec},
    }
    said: list[str] = []
    made = _job.Runner(
        Client(fake.url, "job-key"), job, root, fake.url, say=said.append, heartbeat=0.1
    )
    made.said = said  # type: ignore[attr-defined]
    return made


def test_a_run_job_gets_its_inputs_and_its_results_are_uploaded(
    airon: Airon, tmp_path: Path
) -> None:
    airon.inputs = {
        "data/a.wav": b"aaaa",
        "data/sub/b.wav": b"b" * 3000,  # three parts on the way back
        "models/DY-40/best.pt": b"weights",
    }
    root = tmp_path / "work" / "XP-01-1"
    root.mkdir(parents=True)
    script = (
        'test -f "$AIRON_DATA/a.wav" && test -f "$AIRON_MODELS/DY-40/best.pt" && '
        'mkdir -p "$AIRON_OUTPUT/text" && '
        'cat "$AIRON_DATA/sub/b.wav" > "$AIRON_OUTPUT/text/b.txt" && '
        'echo hi > "$AIRON_OUTPUT/top.txt"'
    )
    made = make(
        airon,
        root,
        {"dry": 'test -d "$AIRON_OUTPUT" || true', "run": script},
        type="run",
        sources={"data": "set", "models": ["DY-40"]},
    )
    assert made.run() == 0
    assert airon.states() == [
        ("clone", "skipped"),
        ("setup", "skipped"),
        ("data", "running"),
        ("data", "passed"),
        ("check", "running"),
        ("check", "passed"),
        ("test", "skipped"),
        ("dry", "running"),
        ("dry", "passed"),  # no checkpoint, no resume: a run job's dry is one short pass
        ("run", "running"),
        ("run", "passed"),
        ("eval", "skipped"),
        ("merge", "skipped"),
        ("finish", "passed"),
    ]
    assert airon.stored == {"text/b.txt": b"b" * 3000, "top.txt": b"hi\n"}
    ran = next(body for stage, state, body in airon.stages if (stage, state) == ("run", "passed"))
    assert ran["detail"]["outputs"] == {"files": 2, "bytes": 3003}
    bars = [line["text"] for line in airon.lines if line["text"].startswith("::airon progress")]
    assert any("name=inputs" in b for b in bars) and any("name=outputs" in b for b in bars)


def test_inputs_already_on_the_disk_are_not_downloaded_again(airon: Airon, tmp_path: Path) -> None:
    airon.inputs = {"data/a.wav": b"aaaa"}
    folder = tmp_path / "inputs"
    client = Client(airon.url, "job-key")
    _io.fetch_inputs(client, folder, lambda _: None)
    _io.fetch_inputs(client, folder, lambda _: None)
    assert airon.downloads == 1 and (folder / "data" / "a.wav").read_bytes() == b"aaaa"


def test_an_input_path_cannot_leave_its_folder(airon: Airon, tmp_path: Path) -> None:
    airon.inputs = {"../escape.txt": b"x"}
    with pytest.raises(_io.TransferFailed, match="outside the folder"):
        _io.fetch_inputs(Client(airon.url, "job-key"), tmp_path / "inputs", lambda _: None)


def test_a_failed_run_uploads_nothing(airon: Airon, tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    made = make(airon, root, {"run": 'echo x > "$AIRON_OUTPUT/x"; exit 3'}, type="run")
    assert made.run() == 1
    assert ("run", "failed") in airon.states() and airon.stored == {}


def test_a_training_job_has_no_run_stage_and_no_output_folder(airon: Airon, tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    made = make(airon, root, {"train": 'test -z "$AIRON_OUTPUT"'})
    assert made.run() == 0
    assert "run" not in [stage for stage, _ in airon.states()]


def git_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    for name, text in files.items():
        (path / name).write_text(text)
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "first")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_setup_comes_from_the_lock_file_and_later_stages_use_it(
    airon: Airon, tmp_path: Path
) -> None:
    commit = git_repo(tmp_path / "origin", {"requirements.txt": "# nothing to install\n"})
    root = tmp_path / "work" / "XP-01-1"
    made = make(airon, root, {"train": 'test "$VIRTUAL_ENV" = "$PWD/.venv" && which python'})
    made.source = {"url": str(tmp_path / "origin"), "commit": commit}
    assert made.run() == 0
    setup = next(
        body for stage, state, body in airon.stages if (stage, state) == ("setup", "running")
    )
    assert setup["detail"] == {"automatic": True} and ("setup", "passed") in airon.states()
    assert (root / ".venv" / "bin" / "python").exists()


def test_the_setup_command_follows_the_lock_file(tmp_path: Path) -> None:
    assert _io.setup_command(tmp_path) is None
    (tmp_path / "pyproject.toml").write_text("")
    assert "pip install --quiet -e ." in (_io.setup_command(tmp_path) or "")
    (tmp_path / "requirements.txt").write_text("")
    assert "-r requirements.txt" in (_io.setup_command(tmp_path) or "")
    (tmp_path / "uv.lock").write_text("")
    assert 'uv" sync --frozen' in (_io.setup_command(tmp_path) or "")


def test_a_retry_on_a_fresh_machine_prepares_the_code_again(airon: Airon, tmp_path: Path) -> None:
    commit = git_repo(tmp_path / "origin", {"marker.txt": "x"})
    root = tmp_path / "work" / "XP-01-1"
    made = make(airon, root, {"test": "exit 1", "train": "cat marker.txt"})
    made.source = {"url": str(tmp_path / "origin"), "commit": commit}
    assert made.run("train") == 0  # retried from train on a machine that has nothing
    states = airon.states()
    assert ("clone", "passed") in states and ("train", "passed") in states
    assert ("test", "skipped") in states  # the work before the start is still skipped


def test_init_writes_a_run_template(tmp_path: Path) -> None:
    from airon import _template

    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "job.py").write_text("")
    text = _template.render(tmp_path, "DY-44", "run")
    assert "type: run" in text and "experiment: DY-44" in text
    assert "run: uv run python job.py" in text and "$AIRON_OUTPUT" in text
    assert "metrics:" not in text  # a run job is not judged by a metric
