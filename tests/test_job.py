"""The job runner (airon run) against a small fake of Airon's job endpoints, with real commands."""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from airon import _job
from airon._transport import Client


class JobServer:
    """Answers /v1/job/* like Airon and records what the machine said."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, str, dict[str, Any]]] = []
        self.lines: list[dict[str, Any]] = []
        self.beats: list[dict[str, Any]] = []
        self.check = {"ok": True, "problems": [], "warnings": []}
        self.dry = {"ok": True, "problems": [], "warnings": [], "checkpoint": None}
        self.commands: list[dict[str, Any]] = []
        self.job: dict[str, Any] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                job = server.job  # read once: a test may hand out a new view per request
                raw = json.dumps(job).encode()
                self.send_response(200 if job else 401)  # no job: the key was revoked
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                answer: dict[str, Any] = {}
                if self.path == "/v1/job/stage":
                    server.stages.append((body["stage"], body["state"], body))
                elif self.path == "/v1/job/logs":
                    server.lines.extend(body["lines"])
                    answer = {"accepted": len(body["lines"])}
                elif self.path == "/v1/job/heartbeat":
                    server.beats.append(body)
                    answer = {"commands": server.commands}
                elif self.path == "/v1/job/check":
                    answer = server.check
                elif self.path == "/v1/job/dry":
                    answer = server.dry
                raw = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def states(self) -> list[tuple[str, str]]:
        return [(stage, state) for stage, state, _ in self.stages]

    def close(self) -> None:
        self.http.shutdown()


@pytest.fixture
def jobs() -> Any:
    server = JobServer()
    yield server
    server.close()


def runner(server: JobServer, root: Path, stages: dict[str, str], **spec: Any) -> _job.Runner:
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
        Client(server.url, "job-key"), job, root, server.url, say=said.append, heartbeat=0.1
    )
    made.said = said  # type: ignore[attr-defined]
    return made


def test_stages_run_in_order_with_the_job_environment(jobs: JobServer, tmp_path: Path) -> None:
    stages = {
        "setup": "echo setup $AIRON_STAGE $AIRON_JOB",
        "train": "echo oops >&2; echo $AIRON_RUN_ID",
    }
    code = runner(jobs, tmp_path, stages).run()
    assert code == 0
    assert jobs.states() == [
        ("clone", "skipped"),
        ("setup", "running"),
        ("setup", "passed"),
        ("data", "skipped"),
        ("check", "running"),
        ("check", "passed"),
        ("test", "skipped"),
        ("dry", "skipped"),
        ("train", "running"),
        ("train", "passed"),
        ("eval", "skipped"),
        ("finish", "passed"),
    ]
    texts = [(line["stream"], line["text"]) for line in jobs.lines]
    assert ("stdout", "setup setup XP-01/1") in texts
    assert ("stderr", "oops") in texts
    assert ("stdout", "00000000-0000-0000-0000-000000000001") in texts


def test_failure_stops_the_job_and_says_why(jobs: JobServer, tmp_path: Path) -> None:
    stages = {"setup": "echo 'No module named torch' >&2; exit 3", "train": "touch trained"}
    assert runner(jobs, tmp_path, stages).run() == 1
    failed = [body for stage, state, body in jobs.stages if state == "failed"]
    assert failed[0]["stage"] == "setup" and failed[0]["exit_code"] == 3
    assert failed[0]["reason"] == "exit code 3; No module named torch"
    assert not (tmp_path / "trained").exists()
    assert "train" not in [stage for stage, _ in jobs.states()]


def test_a_broken_pipe_is_a_failure(jobs: JobServer, tmp_path: Path) -> None:
    # bash -o pipefail: `false | cat` fails instead of passing on cat's exit code.
    assert runner(jobs, tmp_path, {"train": "false | cat; echo after"}).run() == 1


def test_check_problems_stop_before_training(jobs: JobServer, tmp_path: Path) -> None:
    jobs.check = {"ok": False, "problems": ["Expected A10G GPUs"], "warnings": []}
    assert runner(jobs, tmp_path, {"train": "touch trained"}).run() == 1
    assert ("check", "failed") in jobs.states()
    assert not (tmp_path / "trained").exists()
    assert jobs.beats == [] or "gpus" in jobs.beats[0]


def test_dry_resumes_from_the_checkpoint(jobs: JobServer, tmp_path: Path) -> None:
    jobs.dry = {"ok": True, "problems": [], "checkpoint": {"uri": "s3://b/ck", "step": 20}}
    stages = {"dry": "echo ${AIRON_RESUME:-fresh} >> passes", "train": "true"}
    assert runner(jobs, tmp_path, stages).run() == 0
    assert (tmp_path / "passes").read_text().split() == ["fresh", "s3://b/ck"]


def test_dry_problems_fail_the_stage(jobs: JobServer, tmp_path: Path) -> None:
    jobs.dry = {"ok": False, "problems": ["train/loss became NaN"], "checkpoint": None}
    assert runner(jobs, tmp_path, {"dry": "true", "train": "touch trained"}).run() == 1
    failed = [body for _, state, body in jobs.stages if state == "failed"]
    assert failed[0]["reason"] == "train/loss became NaN"
    assert not (tmp_path / "trained").exists()


def test_start_from_a_stage(jobs: JobServer, tmp_path: Path) -> None:
    stages = {"setup": "touch setup", "train": "touch trained"}
    assert runner(jobs, tmp_path, stages).run("train") == 0
    assert ("setup", "skipped") in jobs.states()
    assert not (tmp_path / "setup").exists() and (tmp_path / "trained").exists()


def test_stop_from_airon_kills_the_whole_process_group(jobs: JobServer, tmp_path: Path) -> None:
    jobs.commands = [{"action": "stop"}]
    started = time.monotonic()
    # A child of the shell would survive a signal to the shell alone.
    code = runner(jobs, tmp_path, {"train": "sleep 30 & wait"}).run()
    assert code == 130
    assert time.monotonic() - started < 10
    assert ("train", "killed") in jobs.states()


def test_heartbeat_describes_the_machine(jobs: JobServer, tmp_path: Path) -> None:
    runner(jobs, tmp_path, {"train": "sleep 0.5"}).run()
    beat = jobs.beats[0]
    assert beat["stage"] == "train" and beat["disk_free_gb"] > 0
    assert {"gpus", "last_output_s", "host"} <= set(beat)


def test_pump_collapses_progress_bars() -> None:
    class Ship:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def add(self, stream: str, text: str) -> None:
            self.lines.append(text)

    ship = Ship()
    out = io.BytesIO()
    _job._pump(io.BufferedReader(io.BytesIO(b"10%\r50%\r100%\ndone\npartial")), out, "stdout", ship)  # type: ignore[arg-type]
    assert ship.lines == ["100%", "done", "partial"]
    assert out.getvalue() == b"10%\r50%\r100%\ndone\npartial"


def test_exit_reasons() -> None:
    assert "lack of memory" in _job._exit_reason(137)
    assert "lack of memory" in _job._exit_reason(-9)
    assert _job._exit_reason(-15) == "ended by signal SIGTERM"
    assert "command not found" in _job._exit_reason(127)


def test_fingerprint_follows_input_files(tmp_path: Path) -> None:
    (tmp_path / "requirements.lock").write_text("torch==2.5")
    before = _job.fingerprint("pip install", tmp_path, ["requirements.lock"])
    (tmp_path / "requirements.lock").write_text("torch==2.6")
    assert _job.fingerprint("pip install", tmp_path, ["requirements.lock"]) != before
    assert _job.fingerprint("pip install -U", tmp_path, ["requirements.lock"]) != before


def git_repo(path: Path) -> str:
    """A repository with one commit holding a marker file; returns the commit."""
    import subprocess

    path.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    (path / "marker.txt").write_text("from the commit")
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "first")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_rented_machine_clones_the_commit(jobs: JobServer, tmp_path: Path) -> None:
    commit = git_repo(tmp_path / "origin")
    work = tmp_path / "work" / "XP-01-1"
    made = runner(jobs, work, {"train": "cat marker.txt"})
    made.source = {"url": str(tmp_path / "origin"), "commit": commit}
    assert made.run() == 0
    assert ("clone", "passed") in jobs.states()
    assert ("stdout", "from the commit") in [(x["stream"], x["text"]) for x in jobs.lines]


def test_a_commit_that_is_not_there_says_so(jobs: JobServer, tmp_path: Path) -> None:
    git_repo(tmp_path / "origin")
    made = runner(jobs, tmp_path / "work" / "x", {"train": "touch trained"})
    made.source = {"url": str(tmp_path / "origin"), "commit": "0" * 40}
    assert made.run() == 1
    failed = [body for stage, state, body in jobs.stages if state == "failed"]
    assert failed[0]["stage"] == "clone"
    assert "is the commit pushed" in failed[0]["reason"]


def test_git_access_reads_private_github_without_writing_the_token() -> None:
    url, env = _job.git_access("git@github.com:acme/model.git", "secret")
    assert url == "https://github.com/acme/model.git"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert "secret" not in url
    assert _job.git_access("https://github.com/acme/model", None) == (
        "https://github.com/acme/model",
        {},
    )
    assert _job.git_access("https://gitlab.com/a/b.git", "secret")[1] == {}


def test_agent_fetches_its_job_and_runs_it(
    jobs: JobServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = git_repo(tmp_path / "origin")
    jobs.job = {
        "job": "XP-01/3",
        "run_id": "00000000-0000-0000-0000-000000000003",
        "project": "lab",
        "experiment": "XP-01",
        "repository_url": str(tmp_path / "origin"),
        "commit": commit,
        "spec": {"stages": {"train": "test -f marker.txt && echo $AIRON_JOB"}},
    }
    monkeypatch.setenv("AIRON_URL", jobs.url)
    monkeypatch.setenv("AIRON_API_KEY", "job-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    said: list[str] = []
    assert _job.agent(say=said.append) == 0
    assert (tmp_path / "home" / "work" / "XP-01-3" / "marker.txt").exists()
    assert ("finish", "passed") in jobs.states()
    assert any(line["text"] == "XP-01/3" for line in jobs.lines)


def test_status_watch_prints_changes_and_exits_with_the_result(
    jobs: JobServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from airon import cli

    def view(status: str, stages: list[tuple[str, str]], **extra: Any) -> dict[str, Any]:
        return {
            "job": "XP-01/2",
            "mode": "aws",
            "status": status,
            "commit": "abc",
            "attempts": [
                {
                    "attempt": 1,
                    "status": "failed" if status == "held" else "running",
                    "host": "i-1",
                    "stages": [{"name": n, "state": s} for n, s in stages],
                    "error": {"message": "stage train failed; exit code 137"},
                    "machine": {"up": True, "ssh": "ssh ubuntu@203.0.113.7"},
                }
            ],
            **extra,
        }

    looks = [
        view("queued", [], waiting_for="no g5.xlarge machine right now"),
        view("running", [("clone", "passed"), ("train", "running")]),
        view(
            "held",
            [("clone", "passed"), ("train", "failed")],
            held_until="2026-11-20T10:00:00+00:00",
        ),
    ]

    calls = iter(looks)
    monkeypatch.setattr(
        JobServer, "job", property(lambda self: next(calls, looks[-1])), raising=False
    )
    monkeypatch.setenv("AIRON_URL", jobs.url)
    monkeypatch.setenv("AIRON_API_KEY", "key")
    code = cli.main(["status", "XP-01/2", "--watch", "--every", "0.01"])
    out = capsys.readouterr().out
    assert code == 1
    assert "job queued: no g5.xlarge machine right now" in out
    assert "train running" in out and "train failed" in out
    assert "ssh (for emergencies): ssh ubuntu@203.0.113.7" in out
    assert "kept for a look until" in out


def test_agent_waits_on_a_failure_and_goes_on_with_a_retry(
    jobs: JobServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    origin = tmp_path / "origin"
    first = git_repo(origin)
    (origin / "fixed.txt").write_text("the fix")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fix"],
        cwd=origin,
        check=True,
    )
    fixed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=origin, capture_output=True, text=True, check=True
    ).stdout.strip()

    def job(attempt: int, commit: str, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "job": "XP-01/4",
            "attempt": attempt,
            "status": status,
            "run_id": f"00000000-0000-0000-0000-00000000000{attempt}",
            "project": "lab",
            "experiment": "XP-01",
            "repository_url": str(origin),
            "commit": commit,
            "spec": {
                "stages": {
                    # Made by setup in attempt 1; a retry must find it still there.
                    "setup": "touch .venv-made",
                    "train": "test -f fixed.txt && test -f .venv-made",
                }
            },
            **extra,
        }

    looks = iter(
        [
            job(1, first, "running"),  # the agent fetches its job
            job(1, first, "held"),  # train failed: nothing yet
            job(2, fixed, "starting", from_stage="train"),  # someone retried with the fix
            {},  # after the job is done, the key is gone
        ]
    )
    monkeypatch.setattr(JobServer, "job", property(lambda self: next(looks, {})), raising=False)
    monkeypatch.setenv("AIRON_URL", jobs.url)
    monkeypatch.setenv("AIRON_API_KEY", "job-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    said: list[str] = []
    assert _job.agent(say=said.append, poll=0.05, beat=0.1) == 0
    assert any("retried as attempt 2" in line for line in said)
    states = jobs.states()
    assert ("train", "failed") in states and states[-1] == ("finish", "passed")
    # The retry updated the checkout in place and did not run setup again.
    assert [s for s in states if s[0] == "setup"] == [
        ("setup", "running"),
        ("setup", "passed"),
        ("setup", "skipped"),
    ]


def test_agent_stops_waiting_when_the_machine_is_switched_off(
    jobs: JobServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = git_repo(tmp_path / "origin")
    looks = iter(
        [
            {
                "job": "XP-01/5",
                "attempt": 1,
                "status": "running",
                "run_id": "00000000-0000-0000-0000-000000000005",
                "project": "lab",
                "experiment": "XP-01",
                "repository_url": str(tmp_path / "origin"),
                "commit": commit,
                "spec": {"stages": {"train": "exit 3"}},
            }
        ]
    )
    monkeypatch.setattr(JobServer, "job", property(lambda self: next(looks, {})), raising=False)
    monkeypatch.setenv("AIRON_URL", jobs.url)
    monkeypatch.setenv("AIRON_API_KEY", "job-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _job.agent(say=lambda _: None, poll=0.05, beat=0.1) == 1


def test_init_writes_a_template_that_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from airon import cli

    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--experiment", "DY-40"]) == 0
    text = (tmp_path / "airon.yaml").read_text()
    assert "experiment: DY-40" in text
    assert "setup: pip install -q uv && uv sync --frozen" in text
    assert "train: uv run python scripts/train.py" in text
    assert "dry: uv run python scripts/train.py --max-steps 50" in text
    for lesson in ("FEWER STEPS", "airon.resume()", "airon retry", "Never put secrets"):
        assert lesson in text
    assert cli.main(["init"]) == 2  # never overwrites by surprise
    assert cli.main(["init", "--force"]) == 0
    assert "experiment: CHANGE-ME" in (tmp_path / "airon.yaml").read_text()
