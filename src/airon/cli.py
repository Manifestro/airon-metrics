"""The ``airon`` command: login, logout, doctor, sync, version, and jobs: check, run, status, stop.

Exit codes: 0 success, 1 a check or upload failed, 2 wrong usage or nothing configured.
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import subprocess
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from . import _config, _job, _log
from ._spool import list_spools
from ._sync import AuthError, sync_all
from ._transport import Client, Result
from ._version import PROTOCOL, __version__

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
_PROXY_VARIABLES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def explain(result: Result, url: str) -> str:
    """A plain-language fix for a failed request."""
    parts = urlsplit(url)
    host = parts.hostname or url
    port = parts.port or (443 if parts.scheme == "https" else 80)
    status = result.status
    if result.kind == "dns":
        return (
            f"The host name {host!r} cannot be resolved. Check AIRON_URL for typos and that this "
            "machine has working DNS and internet access."
        )
    if result.kind == "refused":
        return f"Nothing accepts connections at {host}:{port}. Check the port and that the server is running."
    if result.kind == "timeout":
        return (
            f"{host}:{port} did not answer in time. A firewall may block outbound traffic from this "
            "machine, or the address is not reachable from here."
        )
    if result.kind == "tls":
        return (
            f"{result.message}. If the server uses a private certificate authority, point "
            "SSL_CERT_FILE at its certificate bundle. Do not disable verification."
        )
    if result.kind == "proxy":
        return (
            f"The proxy refused the connection ({result.message}). Check HTTPS_PROXY and NO_PROXY, "
            "or unset them if this machine needs no proxy."
        )
    if result.kind == "reset":
        return (
            "The connection was closed unexpectedly. Often the address says http:// where the server "
            "wants https:// (or the other way around), or a proxy is in the way."
        )
    if result.kind == "garbage":
        return (
            "The address answers, but not with JSON. It is probably not an Airon server, or a proxy "
            "or captive portal is answering instead."
        )
    if status == 401:
        return "The server does not accept this key (wrong, revoked, or from another server). Create a key and run `airon login`."
    if status == 403:
        return "The key is valid but may not log runs (read-only or wrong scope). Use a key with tracking access."
    if status == 404:
        return (
            "The address answers but has no /v1/ping, so it is not an Airon server (or AIRON_URL "
            "contains a path that should not be there). Use the server's root address."
        )
    if status is not None and 300 <= status < 400:
        return f"The server redirects ({result.message}). Set AIRON_URL to the final address, usually https://."
    if status == 426:
        return "The server does not speak protocol version 1. Upgrade airon (`pip install -U airon`) or the server."
    if status is not None and status >= 500:
        return "The server reports an internal error. Look at the server's logs, then try again."
    return result.message or "Unexpected answer from the server."


def _proxy_note() -> str:
    names = [v for v in _PROXY_VARIABLES if os.environ.get(v)]
    return f" (a proxy is configured through {', '.join(names)})" if names else ""


def _ok(text: str) -> None:
    print(f"  ok    {text}")


def _bad(text: str, fix: str) -> None:
    print(f"  FAIL  {text}\n        fix: {fix}")


def _key_info(ping: Result) -> dict[str, Any]:
    key = ping.data.get("key")
    return key if isinstance(key, dict) else {}


def cmd_version(_: argparse.Namespace) -> int:
    print(f"airon {__version__} (protocol {PROTOCOL}, python {platform.python_version()})")
    return EXIT_OK


def cmd_login(args: argparse.Namespace) -> int:
    try:
        raw = args.url or input("Server address (for example https://airon.example.com): ")
        key = getpass.getpass("API key (typing is hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return EXIT_USAGE
    url = _config.normalize_url(raw)
    if url is None:
        print(
            f"{raw!r} is not a valid address. Use something like https://airon.example.com",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if not key:
        print("No key entered.", file=sys.stderr)
        return EXIT_USAGE
    client = Client(url, key, timeout=10.0)
    result = client.ping()
    if not result.ok:
        print(f"Could not log in to {url}: {result.message or result.outcome}", file=sys.stderr)
        print(f"  {_log.redact(explain(result, url))}", file=sys.stderr)
        return EXIT_FAILED
    path = _config.save_credentials(url, key)
    info = _key_info(result)
    label = f" as key {info['name']!r}" if info.get("name") else ""
    print(f"Logged in to {url}{label}. Saved to {path} (readable only by you).")
    if urlsplit(url).scheme == "http" and urlsplit(url).hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        print("Note: the address uses plain http, so the key travels unencrypted.")
    return EXIT_OK


def cmd_logout(_: argparse.Namespace) -> int:
    if _config.delete_credentials():
        print(f"Removed {_config.credentials_path()}.")
    else:
        print("No saved credentials.")
    return EXIT_OK


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"airon {__version__}, python {platform.python_version()}\n")
    connection = _config.resolve_connection()
    problems = 0

    if connection.raw_url is None:
        _bad(
            "server address: not set",
            "set AIRON_URL=https://airon.example.com, or run `airon login`",
        )
        problems += 1
    elif connection.url is None:
        _bad(
            f"server address: {connection.raw_url!r} is not a valid http(s) address",
            "use the full address, for example https://airon.example.com",
        )
        problems += 1
    else:
        _ok(f"server address: {connection.url} ({connection.url_source})")
        if urlsplit(connection.url).scheme == "http" and urlsplit(connection.url).hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            print("        note: plain http sends the key unencrypted; prefer https")

    if not connection.api_key:
        _bad("API key: not set", "set AIRON_API_KEY, or run `airon login`")
        problems += 1
    else:
        _ok(f"API key: set ({connection.key_source})")
    if problems:
        return EXIT_USAGE

    assert connection.url and connection.api_key
    client = Client(connection.url, connection.api_key, timeout=10.0)
    started = time.monotonic()
    ping = client.ping()
    millis = round((time.monotonic() - started) * 1000)
    if not ping.ok:
        _bad(
            f"reaching the server: {ping.message or ping.outcome}{_proxy_note()}",
            explain(ping, connection.url),
        )
        return EXIT_FAILED
    _ok(f"reaching the server: answered in {millis} ms")

    key = _key_info(ping)
    if key.get("name"):
        _ok(f"API key accepted: {key.get('name')!r}, scope {key.get('scope', 'unknown')}")
    else:
        _ok("API key accepted")
    protocol = ping.data.get("protocol")
    if protocol != PROTOCOL:
        _bad(
            f"protocol version: server reports {protocol!r}, this SDK speaks {PROTOCOL}",
            "upgrade airon (`pip install -U airon`) or the server so the versions match",
        )
        return EXIT_FAILED
    _ok(f"protocol version: {protocol}")

    failed = _test_run(client, connection.url)
    pending = len(list_spools(_config.spool_root()))
    if pending:
        print(
            f"\n{pending} spooled run(s) are waiting in {_config.spool_root()}; upload with `airon sync`."
        )
    print("\nEverything works." if not failed else "\nSomething needs fixing, see above.")
    return EXIT_FAILED if failed else EXIT_OK


def _test_run(client: Client, url: str) -> bool:
    """Create a tiny run, send two events and finish it. Returns True on failure."""
    run_id = str(uuid.uuid4())
    now = time.time()
    create = {
        "run_id": run_id,
        "project": "airon-doctor",
        "name": f"doctor-{run_id[:6]}",
        "started_at": now,
        "tags": ["doctor"],
    }
    events = [
        {"seq": 1, "t": now, "step": 0, "metrics": {"doctor/ok": 1}},
        {"seq": 2, "t": now, "progress": {"step": 0}},
    ]
    steps: list[tuple[str, Callable[[], Result]]] = [
        ("creating a test run", partial(client.create_run, create)),
        ("sending events", partial(client.send_events, run_id, events)),
        (
            "finishing the run",
            lambda: client.finish_run(run_id, {"status": "finished", "t": time.time()}),
        ),
    ]
    link = ""
    for label, call in steps:
        result = call()
        if not result.ok:
            _bad(f"{label}: {result.message or result.outcome}", explain(result, url))
            return True
        link = str(result.data.get("url") or link)
    _ok(f"test run written and finished: {link or url}")
    return False


def cmd_sync(args: argparse.Namespace) -> int:
    root = _config.spool_root()
    spools = list_spools(root)
    if not spools:
        print(f"Nothing to upload (looked in {root}).")
        return EXIT_OK
    connection = _config.resolve_connection()
    if not connection.complete:
        print(
            "No server configured. Set AIRON_URL and AIRON_API_KEY, or run `airon login`.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    assert connection.url and connection.api_key
    client = Client(connection.url, connection.api_key, timeout=30.0)
    print(f"Uploading {len(spools)} spooled run(s) to {connection.url}")
    try:
        outcomes = sync_all(client, root, force=args.force)
    except AuthError as exc:
        print(f"The server rejected the key ({exc}). Run `airon doctor`.", file=sys.stderr)
        return EXIT_FAILED
    failed = 0
    for outcome in outcomes:
        detail = f"{outcome.events:,} events" if outcome.status == "uploaded" else outcome.message
        print(f"  {outcome.status:<9} {outcome.name}  {detail}")
        failed += outcome.status == "failed"
    uploaded = sum(o.status == "uploaded" for o in outcomes)
    print(f"{uploaded} uploaded, {failed} failed, {len(outcomes) - uploaded - failed} skipped.")
    return EXIT_FAILED if failed else EXIT_OK


# --- jobs (airon.yaml) ----------------------------------------------------------------------------


def _connection_client(timeout: float = 30.0) -> tuple[Client, str] | None:
    connection = _config.resolve_connection()
    if not connection.complete:
        print(
            "No server configured. Set AIRON_URL and AIRON_API_KEY, or run `airon login`.",
            file=sys.stderr,
        )
        return None
    assert connection.url and connection.api_key
    return Client(connection.url, connection.api_key, timeout=timeout), connection.url


def _find_file(name: str | None) -> Path | None:
    if name:
        path = Path(name)
        return path if path.is_file() else None
    here = Path.cwd()
    top = _job.repository(here)["root"]
    for folder in (here, top):
        if (folder / "airon.yaml").is_file():
            return folder / "airon.yaml"
    return None


def _read_job_file(args: argparse.Namespace) -> tuple[Path, str] | None:
    path = _find_file(args.file)
    if path is None:
        print(
            f"No {args.file or 'airon.yaml'} here or at the root of the repository.",
            file=sys.stderr,
        )
        return None
    return path, path.read_text(encoding="utf-8")


def _print_report(report: dict[str, Any]) -> None:
    for error in report.get("errors") or []:
        print(f"  error    {error}")
    for warning in report.get("warnings") or []:
        print(f"  warning  {warning}")


def cmd_check(args: argparse.Namespace) -> int:
    found = _read_job_file(args)
    pair = _connection_client()
    if found is None or pair is None:
        return EXIT_USAGE
    path, text = found
    client, _ = pair
    body = {"yaml": text, "mode": "local" if args.local else "aws", "experiment": args.experiment}
    result = client.request("POST", "/v1/jobs/check", body)
    if not result.ok:
        print(f"Could not check {path}: {_job._plain(result)}", file=sys.stderr)
        return EXIT_FAILED
    report = result.data or {}
    experiment = report.get("experiment")
    print(f"{path}" + (f" for {experiment['code']} ({experiment['title']})" if experiment else ""))
    _print_report(report)
    print("  ok, ready to run" if report.get("ok") else "  not ready, see above")
    return EXIT_OK if report.get("ok") else EXIT_FAILED


def cmd_run(args: argparse.Namespace) -> int:
    found = _read_job_file(args)
    pair = _connection_client()
    if found is None or pair is None:
        return EXIT_USAGE
    path, text = found
    client, url = pair
    repo = _job.repository(path.parent.resolve())
    if repo.get("dirty"):
        print("airon: note: the checkout has uncommitted changes; the run is marked dirty.")
    body = {
        "yaml": text,
        "mode": "local",
        "experiment": args.experiment,
        "from_stage": args.from_stage,
        "commit": repo.get("commit"),
        "dirty": repo.get("dirty", False),
        "repository_url": repo.get("remote"),
        "host": platform.node(),
    }
    result = client.request("POST", "/v1/jobs", body)
    if not result.ok:
        print(f"airon: cannot start: {_job._plain(result)}", file=sys.stderr)
        return EXIT_FAILED
    job = result.data or {}
    for warning in job.get("warnings") or []:
        print(f"airon: warning: {warning}")
    print(f"airon: {job['job']} started, {job.get('url')}")
    runner = _job.Runner(Client(url, job["token"], timeout=30.0), job, repo["root"], url)
    return runner.run(args.from_stage)


def _pushed(root: Path) -> bool:
    """True when HEAD is on some remote branch, so a new machine can fetch it."""
    try:
        out = subprocess.run(
            ["git", "branch", "-r", "--contains", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def cmd_launch(args: argparse.Namespace) -> int:
    found = _read_job_file(args)
    pair = _connection_client(timeout=120.0)
    if found is None or pair is None:
        return EXIT_USAGE
    path, text = found
    client, _ = pair
    repo = _job.repository(path.parent.resolve())
    if not repo.get("commit") or not repo.get("remote"):
        print("airon: launch needs a git checkout with a remote (origin).", file=sys.stderr)
        return EXIT_USAGE
    if not _pushed(repo["root"]):
        print(
            f"airon: commit {repo['commit'][:12]} is not on any remote branch; push it first, "
            "the new machine clones it from there.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    if repo.get("dirty"):
        print("airon: note: uncommitted changes stay here; the machine runs the commit.")
    body = {
        "yaml": text,
        "mode": "aws",
        "experiment": args.experiment,
        "commit": repo["commit"],
        "repository_url": repo["remote"],
    }
    result = client.request("POST", "/v1/jobs", body)
    if not result.ok:
        print(f"airon: cannot launch: {_job._plain(result)}", file=sys.stderr)
        return EXIT_FAILED
    job = result.data or {}
    for warning in job.get("warnings") or []:
        print(f"airon: warning: {warning}")
    machine = job.get("machine") or {}
    price = machine.get("price_per_hr")
    print(
        f"airon: {job['job']} launched on {machine.get('gpu') or 'a machine'} "
        f"({machine.get('pod_id')}"
        + (f", ${price:.2f}/h" if isinstance(price, (int, float)) else "")
        + f"), {job.get('url')}"
    )
    print(
        f"airon: follow it with `airon status {job['job']}`; stop it with `airon stop {job['job']}`."
    )
    return EXIT_OK


def cmd_agent(_: argparse.Namespace) -> int:
    return _job.agent()


def cmd_status(args: argparse.Namespace) -> int:
    pair = _connection_client()
    if pair is None:
        return EXIT_USAGE
    client, _ = pair
    result = client.request("GET", f"/v1/jobs/{args.job}/status")
    if not result.ok:
        print(f"airon: {_job._plain(result)}", file=sys.stderr)
        return EXIT_FAILED
    job = result.data or {}
    print(f"{job['job']}  {job['status']}  ({job['mode']}, commit {job.get('commit') or '-'})")
    for attempt in job.get("attempts") or []:
        dirty = " dirty" if attempt.get("dirty") else ""
        print(
            f"  attempt {attempt['attempt']}: {attempt['status']}{dirty} on {attempt.get('host') or '-'}"
        )
        for stage in attempt.get("stages") or []:
            why = f"  {stage['reason']}" if stage.get("reason") else ""
            print(f"    {stage['name']:<7} {stage['state']}{why}")
        error = attempt.get("error")
        if error and attempt["status"] != "running":
            print(f"  cause: {error.get('message')}")
    return EXIT_OK


def cmd_stop(args: argparse.Namespace) -> int:
    pair = _connection_client()
    if pair is None:
        return EXIT_USAGE
    client, _ = pair
    result = client.request("POST", f"/v1/jobs/{args.job}/stop", {})
    if not result.ok:
        print(f"airon: {_job._plain(result)}", file=sys.stderr)
        return EXIT_FAILED
    print(f"{args.job}: stopping; the machine hears it within 30 seconds.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airon",
        description="Command line helpers for the Airon experiment tracker.",
        epilog="Exit codes: 0 success, 1 a check or upload failed, 2 wrong usage or nothing configured.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command")
    login = commands.add_parser("login", help="save the server address and API key")
    login.add_argument("--url", help="server address (asked for if omitted)")
    login.set_defaults(func=cmd_login)
    commands.add_parser("logout", help="forget the saved credentials").set_defaults(func=cmd_logout)
    commands.add_parser("doctor", help="check the setup and write a test run").set_defaults(
        func=cmd_doctor
    )
    sync = commands.add_parser("sync", help="upload runs saved while offline")
    sync.add_argument(
        "--force",
        action="store_true",
        help="also upload runs that look like they are still running",
    )
    sync.set_defaults(func=cmd_sync)
    commands.add_parser("version", help="print the version").set_defaults(func=cmd_version)

    check = commands.add_parser("check", help="validate airon.yaml with the server, run nothing")
    check.add_argument("file", nargs="?", help="path to the file (default: airon.yaml)")
    check.add_argument("--experiment", help="experiment id, if airon.yaml does not name one")
    check.add_argument("--local", action="store_true", help="check for `airon run`, not launch")
    check.set_defaults(func=cmd_check)
    run = commands.add_parser("run", help="run airon.yaml here, stage by stage, reporting to Airon")
    run.add_argument("file", nargs="?", help="path to the file (default: airon.yaml)")
    run.add_argument("--experiment", help="experiment id, if airon.yaml does not name one")
    run.add_argument("--from", dest="from_stage", choices=_job.STAGES, help="start at this stage")
    run.set_defaults(func=cmd_run)
    launch = commands.add_parser(
        "launch", help="run airon.yaml on a machine Airon rents (billable); needs a pushed commit"
    )
    launch.add_argument("file", nargs="?", help="path to the file (default: airon.yaml)")
    launch.add_argument("--experiment", help="experiment id, if airon.yaml does not name one")
    launch.set_defaults(func=cmd_launch)
    commands.add_parser(
        "agent", help="run this machine's job (started by the launch, not by people)"
    ).set_defaults(func=cmd_agent)
    status = commands.add_parser("status", help="show a job: DY-40 (its latest) or DY-40/2")
    status.add_argument("job")
    status.set_defaults(func=cmd_status)
    stop = commands.add_parser("stop", help="stop a job")
    stop.add_argument("job")
    stop.set_defaults(func=cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_FAILED
