"""Running an airon.yaml: the stages one after another, each reported to Airon.

`airon run` uses this on any machine (a laptop, an own server); the machine agent on a rented
machine will use the same code. The server parses and validates the file; this side executes it.
See the server's docs/jobs.md for the contract.
"""

from __future__ import annotations

import base64
import hashlib
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Callable

from . import _log
from ._system import GpuReader, HostReader
from ._transport import Client, Result

STAGES = ("clone", "setup", "data", "check", "test", "dry", "train", "eval", "finish")
HEARTBEAT = 30.0
LOG_FLUSH = 2.0
LOG_BATCH = 500
STOP_GRACE = 30.0

Say = Callable[[str], None]


def _say(text: str) -> None:
    # Flushed at once: the stage commands write straight to the terminal, and our lines must
    # appear in order with theirs.
    print(text, flush=True)


class JobFailed(Exception):
    """Something before the stages could not be done (no server, a refused file...)."""


def _plain(result: Result) -> str:
    data = result.data or {}
    return str(data.get("error") or result.message or result.outcome)


# --- git ------------------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def repository(start: Path) -> dict[str, Any]:
    """Where the code is: the root of the git checkout (or the folder), commit, dirty, remote."""
    top = _git(start, "rev-parse", "--show-toplevel")
    root = Path(top) if top else start
    info: dict[str, Any] = {"root": root}
    if top:
        info["commit"] = _git(root, "rev-parse", "HEAD")
        info["dirty"] = bool(_git(root, "status", "--porcelain"))
        remote = _git(root, "remote", "get-url", "origin")
        if remote:
            # Credentials in a remote URL never leave the machine.
            info["remote"] = (
                remote.split("@", 1)[-1] if "://" in remote and "@" in remote else remote
            )
    return info


def fingerprint(command: str, root: Path, files: list[str]) -> str:
    """The stage's command and the content of its input files: same fingerprint, same result."""
    digest = hashlib.sha256(command.encode())
    for name in sorted(files):
        path = root / name
        digest.update(f"\0{name}\0".encode())
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:32]


# --- shipping console lines -----------------------------------------------------------------------


class LogShipper:
    """Collects output lines and sends them to the job's run in batches, off the hot path."""

    def __init__(self, client: Client):
        self._client = client
        self._lines: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="airon-job-logs", daemon=True)
        self.last_output = time.time()
        # The last line a command wrote to stderr: usually the line that says why it failed.
        self.last_error = ""

    def start(self) -> None:
        self._thread.start()

    def add(self, stream: str, text: str) -> None:
        self.last_output = time.time()
        if stream == "stderr" and text.strip():
            self.last_error = text.strip()[:500]
        with self._lock:
            self._lines.append({"t": self.last_output, "stream": stream, "text": text})
            if len(self._lines) > 20_000:  # the server keeps the tail anyway
                del self._lines[:5_000]
            if len(self._lines) >= LOG_BATCH:
                self._wake.set()

    def flush(self) -> None:
        while True:
            with self._lock:
                batch, self._lines = self._lines[:LOG_BATCH], self._lines[LOG_BATCH:]
            if not batch:
                return
            result = self._client.request("POST", "/v1/job/logs", {"lines": batch}, timeout=15)
            if not result.ok:
                _log.warn_once("job-logs", f"could not send console lines: {_plain(result)}")
                return

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)
        self.flush()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(LOG_FLUSH)
            self._wake.clear()
            try:
                self.flush()
            except Exception:
                pass


def _pump(source: IO[bytes], sink: IO[bytes], stream: str, ship: LogShipper) -> None:
    """Copies a child's output to our own terminal as it comes (progress bars keep working) and
    hands complete lines, with carriage-return redraws collapsed, to the shipper."""
    pending = b""
    while True:
        chunk = source.read1(65536) if hasattr(source, "read1") else source.read(65536)
        if not chunk:
            break
        try:
            sink.write(chunk)
            sink.flush()
        except (OSError, ValueError):
            pass
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            ship.add(stream, line.rsplit(b"\r", 1)[-1].decode("utf-8", "replace").rstrip())
    if pending.strip():
        ship.add(stream, pending.rsplit(b"\r", 1)[-1].decode("utf-8", "replace").rstrip())


# --- the runner -----------------------------------------------------------------------------------


class Runner:
    """Executes one job on this machine. `run()` returns the process exit code."""

    def __init__(
        self,
        client: Client,
        job: dict[str, Any],
        root: Path,
        server_url: str,
        say: Say = _say,
        heartbeat: float = HEARTBEAT,
        source: dict[str, str] | None = None,
    ):
        self.client = client
        # On a machine Airon rented, where to clone the code from: {"url", "commit"}.
        self.source = source
        self.job = job
        self.spec: dict[str, Any] = job["spec"]
        self.root = root
        self.url = server_url
        self.say = say
        self.heartbeat_interval = heartbeat
        self.ship = LogShipper(client)
        self.child: subprocess.Popen[bytes] | None = None
        self.stopping = threading.Event()
        self.stage: str | None = None
        self._beat = threading.Thread(target=self._heartbeat, name="airon-job-beat", daemon=True)
        self._done = threading.Event()
        self._gpus = GpuReader()
        self._host = HostReader(str(root))

    # -- talking to Airon --

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        result = self.client.request("POST", path, body, timeout=30)
        if not result.ok:
            raise JobFailed(f"Airon refused {path}: {_plain(result)}")
        return result.data or {}

    def report(self, stage: str, state: str, **extra: Any) -> None:
        self.ship.flush()  # the lines that explain a failure arrive before the failure
        body = {"stage": stage, "state": state, **{k: v for k, v in extra.items() if v is not None}}
        result = self.client.request("POST", "/v1/job/stage", body, timeout=30)
        if not result.ok and result.status not in (401, 409):
            _log.warn_once(f"stage-{stage}", f"could not report {stage} {state}: {_plain(result)}")

    def machine(self) -> dict[str, Any]:
        gpus = [
            {
                "index": g.get("index"),
                "name": g.get("name"),
                "util_pct": g.get("util"),
                "mem_used_mb": g.get("mem_used"),
                "mem_total_mb": g.get("mem_total"),
            }
            for g in self._gpus.read()
        ]
        host = self._host.read()
        usage = shutil.disk_usage(self.root if self.root.exists() else self.root.parent)
        return {
            "host": platform.node(),
            "stage": self.stage,
            "gpus": gpus,
            "cpu_util_pct": host.get("cpu/util_pct"),
            "ram_used_mb": host.get("ram/used_mb"),
            "ram_total_mb": host.get("ram/total_mb"),
            "disk_free_gb": round(usage.free / 1e9, 1),
            "last_output_s": round(time.time() - self.ship.last_output, 1),
            "child_pid": self.child.pid if self.child else None,
        }

    def _heartbeat(self) -> None:
        while not self._done.wait(self.heartbeat_interval):
            try:
                result = self.client.request(
                    "POST", "/v1/job/heartbeat", self.machine(), timeout=15
                )
            except Exception:
                continue
            if result.status == 401 and not self._done.is_set():
                # The key was revoked: the job was ended elsewhere.
                self.stop("the job was ended in Airon")
                continue
            for command in (result.data or {}).get("commands") or []:
                if command.get("action") == "stop":
                    self.stop("stopped from Airon")

    # -- processes --

    def stop(self, why: str) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.say(f"airon: stopping ({why})")
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

        def kill_later() -> None:
            try:
                child.wait(STOP_GRACE)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        threading.Thread(target=kill_later, daemon=True).start()

    def environment(self, stage: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env") or {})
        env.update(
            {
                "AIRON_URL": self.url,
                "AIRON_API_KEY": self.job["token"],
                "AIRON_RUN_ID": self.job["run_id"],
                "AIRON_PROJECT": self.job["project"],
                "AIRON_EXPERIMENT": self.job["experiment"],
                "AIRON_JOB": self.job["job"],
                "AIRON_STAGE": stage,
                "AIRON_MODE": "online",
            }
        )
        env.pop("AIRON_RESUME", None)
        # The runner announces every stage; the SDK's own "tracking run" lines would repeat it.
        # Its warnings still show.
        env.setdefault("AIRON_QUIET", "1")
        env.update(extra or {})
        return env

    def execute(
        self,
        stage: str,
        command: str,
        extra: dict[str, str] | None = None,
        cwd: Path | None = None,
        shown: str | None = None,
    ) -> int:
        """Runs one shell command of a stage and returns its exit code (negative: a signal)."""
        self.say(f"airon: {stage}: {shown or command}")
        child = subprocess.Popen(
            ["bash", "-eo", "pipefail", "-c", command],
            cwd=cwd or self.root,
            env=self.environment(stage, extra),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # its own process group, so a stop reaches every child
        )
        self.child = child
        self.ship.last_error = ""
        assert child.stdout is not None and child.stderr is not None
        pumps = [
            threading.Thread(
                target=_pump,
                args=(child.stdout, sys.stdout.buffer, "stdout", self.ship),
                daemon=True,
            ),
            threading.Thread(
                target=_pump,
                args=(child.stderr, sys.stderr.buffer, "stderr", self.ship),
                daemon=True,
            ),
        ]
        for pump in pumps:
            pump.start()
        code = child.wait()
        for pump in pumps:
            pump.join(timeout=5)
        self.child = None
        return code

    # -- stages --

    def run(self, from_stage: str | None = None) -> int:
        self.ship.start()
        self._beat.start()
        previous = {
            sig: signal.signal(sig, lambda *_: self.stop("interrupted on this machine"))
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return self._stages(from_stage)
        finally:
            self._done.set()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.ship.close()

    def _stages(self, from_stage: str | None) -> int:
        start = STAGES.index(from_stage) if from_stage else 0
        commands: dict[str, str] = self.spec.get("stages") or {}
        for index, stage in enumerate(STAGES):
            self.stage = stage
            if self.stopping.is_set():
                self.report(stage, "killed", reason="stopped before this stage")
                return 130
            # On a retry the code must be the new commit even when an earlier stage is the start.
            update = stage == "clone" and self.source and (self.root / ".git").exists()
            if index < start and not update:
                if stage in commands or stage in ("clone", "check"):
                    self.report(stage, "skipped", reason=f"started from {from_stage}")
                continue
            began = time.monotonic()
            ok = self._stage(stage, commands)
            if ok is None:
                continue
            if not ok:
                return 130 if self.stopping.is_set() else 1
            if stage != "finish":
                self.say(f"airon: {stage} passed in {_span(time.monotonic() - began)}")
        self.say(f"airon: {self.job['job']} done")
        return 0

    def _stage(self, stage: str, commands: dict[str, str]) -> bool | None:
        """True passed, False failed or stopped, None skipped."""
        if stage == "clone":
            if self.source:
                return self._clone(self.source)
            self.report(
                stage, "skipped", reason="airon run uses this checkout as it is", detail=self._git()
            )
            return None
        if stage == "check":
            return self._check()
        if stage == "finish":
            self.report(stage, "passed")
            return True
        command = commands.get(stage)
        if command is None:
            self.report(stage, "skipped", reason="not in airon.yaml")
            return None
        inputs = (self.spec.get("inputs") or {}).get(stage) or []
        self.report(stage, "running", fingerprint=fingerprint(command, self.root, inputs))
        if stage == "dry":
            return self._dry(command)
        code = self.execute(stage, command)
        return self._ended(stage, code)

    def _ended(self, stage: str, code: int, reason: str | None = None) -> bool:
        if self.stopping.is_set():
            self.report(stage, "killed", exit_code=code, reason="stopped")
            return False
        if code != 0:
            why = reason or _exit_reason(code)
            if self.ship.last_error and self.ship.last_error not in why:
                why = f"{why}; {self.ship.last_error}"
            self.say(f"airon: {stage} failed: {why}")
            self.report(stage, "failed", exit_code=code, reason=why)
            return False
        self.report(stage, "passed", exit_code=0)
        return True

    def _clone(self, source: dict[str, str]) -> bool:
        """Clone the repository at the job's commit into the work folder. A GitHub token from the
        pod's secret (GH_TOKEN) is passed to git through its environment, so it is neither shown
        nor written into the checkout."""
        url, commit = source.get("url") or "", source.get("commit") or ""
        self.report("clone", "running", detail={"url": url, "commit": commit})
        if not url or not commit:
            self.report("clone", "failed", reason="the job names no repository or commit")
            return False
        url, extra = git_access(url, os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
        self.root.parent.mkdir(parents=True, exist_ok=True)
        here = _q(str(self.root))
        if (self.root / ".git").exists():
            # A retry on the same machine: bring the checkout to the new commit and keep what the
            # earlier attempt left (virtualenv, downloaded data), so nothing is set up again.
            command = (
                f"git -C {here} fetch --quiet {_q(url)} '+refs/heads/*:refs/remotes/origin/*' && "
                f"git -C {here} checkout --quiet --detach {_q(commit)}"
            )
            shown = f"git fetch {url} @ {commit}"
        else:
            if self.root.exists():
                shutil.rmtree(self.root)
            command = (
                f"git clone --quiet --no-checkout {_q(url)} {here} && "
                f"git -C {here} checkout --quiet --detach {_q(commit)}"
            )
            shown = f"git clone {url} @ {commit}"
        code = self.execute("clone", command, extra, cwd=self.root.parent, shown=shown)
        if code != 0:
            hint = (
                "; is the commit pushed, and may GH_TOKEN in the pod secret read the repository?"
                if code == 128
                else ""
            )
            return self._ended("clone", code, _exit_reason(code) + hint)
        return self._ended("clone", 0)

    def _git(self) -> dict[str, Any]:
        info = repository(self.root)
        return {k: v for k, v in info.items() if k in ("commit", "dirty", "remote")}

    def _check(self) -> bool:
        self.report("check", "running")
        facts = self.machine()
        try:
            answer = self._post("/v1/job/check", facts)
        except JobFailed as exc:
            self.report("check", "failed", reason=str(exc))
            return False
        for warning in answer.get("warnings") or []:
            self.say(f"airon: check: {warning}")
        if not answer.get("ok"):
            problems = answer.get("problems") or []
            for problem in problems:
                self.say(f"airon: check: {problem}")
            self.report("check", "failed", reason=" ".join(problems), detail={"machine": facts})
            return False
        self.report("check", "passed", detail={"gpus": [g.get("name") for g in facts["gpus"]]})
        return True

    def _dry(self, command: str) -> bool:
        code = self.execute("dry", command)
        if code != 0 or self.stopping.is_set():
            return self._ended("dry", code)
        first = self._post("/v1/job/dry", {"phase": "first"})
        for warning in first.get("warnings") or []:
            self.say(f"airon: dry: {warning}")
        if not first.get("ok"):
            return self._dry_failed(first.get("problems") or [])
        detail = {k: first.get(k) for k in ("steps_per_sec", "eta_seconds", "checkpoint")}
        checkpoint = first.get("checkpoint")
        if checkpoint and checkpoint.get("uri"):
            self.say(f"airon: dry: continuing from the checkpoint {checkpoint['uri']}")
            since = time.time()
            code = self.execute("dry", command, {"AIRON_RESUME": str(checkpoint["uri"])})
            if code != 0 or self.stopping.is_set():
                return self._ended("dry", code, "the run from the checkpoint " + _exit_reason(code))
            step = checkpoint.get("step")
            again = self._post(
                "/v1/job/dry",
                {"phase": "resume", "since": since, "from_step": step if step is not None else -1},
            )
            if not again.get("ok"):
                return self._dry_failed(again.get("problems") or [])
            detail["resumed_at_step"] = again.get("first_step")
        eta = first.get("eta_seconds")
        if eta:
            self.say(f"airon: dry: the full train should take about {_span(eta)}")
        self.report("dry", "passed", exit_code=0, detail=detail)
        return True

    def _dry_failed(self, problems: list[str]) -> bool:
        for problem in problems:
            self.say(f"airon: dry: {problem}")
        self.report("dry", "failed", reason=" ".join(problems) or "the dry run's checks failed")
        return False


def _q(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def git_access(url: str, token: str | None) -> tuple[str, dict[str, str]]:
    """The URL to clone and the environment that lets git read a private GitHub repository: an
    SSH address becomes HTTPS (a fresh machine has no SSH key for GitHub), and the token goes in
    an extra header through git's environment-only configuration."""
    found = re.match(r"^(?:ssh://)?git@github\.com[:/](.+?)(?:\.git)?/?$", url)
    if found:
        url = f"https://github.com/{found.group(1)}.git"
    if not token or not url.startswith("https://github.com/"):
        return url, {}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return url, {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
        "GIT_TERMINAL_PROMPT": "0",
    }


def agent(say: Say = _say, poll: float = 15.0, beat: float = HEARTBEAT) -> int:
    """`airon agent`: on a machine Airon rented, fetch this machine's job and run it. The job key
    and the server's address come from the environment the launch set up."""
    from . import _config

    connection = _config.resolve_connection()
    if not connection.complete:
        say("airon agent: AIRON_URL and AIRON_API_KEY (the job key) are needed")
        return 2
    assert connection.url and connection.api_key
    client = Client(connection.url, connection.api_key, timeout=30.0)
    result = Result("retry")
    for attempt in range(10):  # the network may come up a little after the service
        result = client.request("GET", "/v1/job", timeout=30)
        if result.ok or result.status in (401, 403):
            break
        time.sleep(min(30, 2**attempt))
    if not result.ok:
        say(f"airon agent: cannot fetch the job: {_plain(result)}")
        return 1
    job = dict(result.data or {})
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(job.get("job") or "job"))
    root = Path.home() / "work" / label
    while True:
        job["token"] = connection.api_key
        say(f"airon agent: running {job.get('job')} attempt {job.get('attempt')} in {root}")
        source = {
            "url": str(job.get("repository_url") or ""),
            "commit": str(job.get("commit") or ""),
        }
        runner = Runner(client, job, root, connection.url, say=say, source=source, heartbeat=beat)
        code = runner.run(job.get("from_stage"))
        if code == 0 or runner.stopping.is_set():
            return code
        say(
            "airon agent: the job failed; this machine waits here for a retry until it is "
            "switched off"
        )
        retried = _wait_for_retry(client, job.get("attempt"), poll, say)
        if retried is None:
            return code
        job = retried


def _wait_for_retry(client: Client, attempt: Any, poll: float, say: Say) -> dict[str, Any] | None:
    """While the job is held after a failure, ask Airon now and then whether someone retried it.
    Returns the new attempt's job, or None once the job is over or the key is gone (the machine is
    being switched off)."""
    while True:
        time.sleep(poll)
        result = client.request("GET", "/v1/job", timeout=30)
        if result.status in (401, 403):
            return None
        if not result.ok:
            continue
        job = dict(result.data or {})
        if job.get("status") in ("done", "failed", "killed"):
            return None
        if job.get("attempt") != attempt and job.get("status") in ("starting", "running"):
            say(f"airon agent: retried as attempt {job.get('attempt')}")
            return job


def _exit_reason(code: int) -> str:
    if code in (137, -9):
        return f"exit code {code}: killed, most likely for lack of memory"
    if code < 0:
        try:
            return f"ended by signal {signal.Signals(-code).name}"
        except ValueError:
            return f"ended by signal {-code}"
    if code == 127:
        return "exit code 127: command not found"
    return f"exit code {code}"


def _span(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {s % 3600 // 60}m"
