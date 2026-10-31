"""The ``airon`` command: login, logout, doctor, sync, version.

Exit codes: 0 success, 1 a check or upload failed, 2 wrong usage or nothing configured.
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import sys
import time
import uuid
from functools import partial
from typing import Any, Callable
from urllib.parse import urlsplit

from . import _config, _log
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
