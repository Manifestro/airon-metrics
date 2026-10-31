from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import airon  # noqa: E402
from airon import _hooks, _log, _run  # noqa: E402
from fake_server import FakeServer  # noqa: E402

KEY = "test-key"
FAST: dict[str, Any] = {
    "flush_interval": 0.05,
    "heartbeat_interval": 30.0,
    "system_interval": 5.0,
    "retry_min_delay": 0.02,
    "retry_max_delay": 0.2,
    "request_timeout": 2.0,
    "shutdown_timeout": 5.0,
}


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """No test may see the developer's real settings, spool or proxies."""
    for name in list(os.environ):
        if name.startswith("AIRON_") or name in (
            "RANK",
            "LOCAL_RANK",
            "RUNPOD_POD_ID",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AIRON_DIR", str(tmp_path / "airon"))
    monkeypatch.setenv("AIRON_QUIET", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _log.reset()
    yield tmp_path
    for run in _hooks.active():
        run.finish()
    _run._current = None


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer().start()
    yield fake
    fake.stop()


@pytest.fixture
def start(server: FakeServer) -> Callable[..., airon.Run]:
    """``start(**overrides)`` begins a fast, quiet run against the fake server."""

    def make(**overrides: Any) -> airon.Run:
        options: dict[str, Any] = {
            "project": "proj",
            "url": server.url,
            "api_key": KEY,
            "console": False,
            "system_metrics": False,
            **FAST,
        }
        options.update(overrides)
        return airon.init(**options)

    return make


@pytest.fixture
def spool_root(isolated_environment: Path) -> Path:
    return isolated_environment / "airon" / "runs"
