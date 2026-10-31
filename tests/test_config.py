from __future__ import annotations

import json
import os
import re
import stat
import threading
from pathlib import Path

import pytest

import airon
from airon import _config
from fake_server import FakeServer


def configure(monkeypatch, server):
    monkeypatch.setenv("AIRON_URL", server.url)
    monkeypatch.setenv("AIRON_API_KEY", "test-key")


# ---- where settings come from -----------------------------------------------------------------


def test_nothing_is_configured_by_default():
    connection = _config.resolve_connection()
    assert connection.url is None and connection.api_key is None and not connection.complete


def test_arguments_beat_environment_beats_credentials_file(monkeypatch):
    _config.save_credentials("https://file.example.com", "file-key-000")
    assert _config.resolve_connection().url == "https://file.example.com"
    assert _config.resolve_connection().api_key == "file-key-000"
    monkeypatch.setenv("AIRON_URL", "https://env.example.com/")
    monkeypatch.setenv("AIRON_API_KEY", "env-key-000")
    connection = _config.resolve_connection()
    assert (connection.url, connection.api_key) == ("https://env.example.com", "env-key-000")
    assert "AIRON_URL" in connection.url_source
    connection = _config.resolve_connection("https://arg.example.com", "arg-key-000")
    assert (connection.url, connection.api_key) == ("https://arg.example.com", "arg-key-000")


def test_precedence_decides_where_events_go(server, start, monkeypatch):
    other = FakeServer(keys=("other-key",)).start()
    try:
        _config.save_credentials(other.url, "other-key")
        run = airon.init(project="p", console=False, system_metrics=False, flush_interval=0.05)
        run.log({"x": 1})
        run.finish()
        assert len(other.runs) == 1 and not server.runs  # from the file

        configure(monkeypatch, server)
        run = airon.init(project="p", console=False, system_metrics=False, flush_interval=0.05)
        run.log({"x": 1})
        run.finish()
        assert len(server.runs) == 1  # the environment beat the file
    finally:
        other.stop()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://airon.example.com", "https://airon.example.com"),
        ("https://airon.example.com/", "https://airon.example.com"),
        ("airon.example.com", "https://airon.example.com"),
        ("http://localhost:8080/", "http://localhost:8080"),
        ("https://example.com/prefix/", "https://example.com/prefix"),
        ("  https://example.com  ", "https://example.com"),
        ("ftp://example.com", None),
        ("https://", None),
        ("http://host:notaport", None),
        ("", None),
    ],
)
def test_normalize_url(raw, expected):
    assert _config.normalize_url(raw) == expected


def test_credentials_file_permissions(tmp_path, monkeypatch):
    target = tmp_path / "fresh" / "dir"
    monkeypatch.setenv("AIRON_DIR", str(target))
    path = _config.save_credentials("https://airon.example.com", "k-123456789")
    assert path == target / "credentials.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert json.loads(path.read_text()) == {
        "url": "https://airon.example.com",
        "api_key": "k-123456789",
    }
    _config.save_credentials("https://other.example.com", "k-987654321")  # overwrite keeps the mode
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(target.glob("*.tmp"))


def test_default_credentials_location_is_under_the_config_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRON_DIR")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    path = _config.save_credentials("https://airon.example.com", "k-123456789")
    assert path == tmp_path / "h" / ".config" / "airon" / "credentials.json"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert _config.spool_root() == tmp_path / "h" / ".cache" / "airon" / "runs"


def test_corrupt_credentials_file_is_ignored(tmp_path):
    path = _config.credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert _config.load_credentials() == {}
    path.write_text('["a list"]')
    assert _config.load_credentials() == {}


def test_source_contains_no_server_addresses():
    """The SDK has no default server and the repository is public."""
    root = Path(airon.__file__).parent
    allowed = ("example.com", "localhost", "127.0.0.1", "github.com/Manifestro")
    for path in root.rglob("*.py"):
        for url in re.findall(r"https?://[A-Za-z0-9][^\s\"')`>]*", path.read_text()):
            assert any(a in url for a in allowed), f"{path.name}: {url}"
        assert not re.search(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", path.read_text().replace("127.0.0.1", "")
        )


# ---- modes ----------------------------------------------------------------------------------


def test_env_mode_offline(server, monkeypatch, spool_root):
    configure(monkeypatch, server)
    monkeypatch.setenv("AIRON_MODE", "offline")
    run = airon.init(project="p", console=False, system_metrics=False)
    run.log({"x": 1})
    run.finish()
    assert run.mode == "offline" and server.requests == []
    assert any(spool_root.iterdir())


def test_env_mode_online_is_the_default(server, monkeypatch):
    configure(monkeypatch, server)
    monkeypatch.setenv("AIRON_MODE", "ONLINE")
    run = airon.init(project="p", console=False, system_metrics=False, flush_interval=0.05)
    run.finish()
    assert run.mode == "online" and server.runs


def test_argument_beats_env_mode(server, monkeypatch):
    configure(monkeypatch, server)
    monkeypatch.setenv("AIRON_MODE", "disabled")
    run = airon.init(project="p", mode="online", console=False, system_metrics=False)
    run.finish()
    assert run.mode == "online" and server.runs


def test_unknown_mode_warns_and_uses_online(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    monkeypatch.setenv("AIRON_MODE", "turbo")
    run = airon.init(project="p", console=False, system_metrics=False)
    run.finish()
    assert run.mode == "online"
    assert "unknown mode 'turbo'" in capsys.readouterr().err


def test_disabled_is_a_cheap_complete_no_op(server, monkeypatch, tmp_path):
    configure(monkeypatch, server)
    monkeypatch.setenv("AIRON_MODE", "disabled")
    threads_before = threading.active_count()
    with airon.init(project="p", total_steps=10) as run:
        run.log({"x": 1}, step=1, epoch=0.1, samples=4, tokens=8)
        run.progress(step=2, total_steps=3)
        run.checkpoint("s3://x", label="best", loss=1.0)
        run.config["lr"] = 1
        run.config.update(a=1)
        run.summary["best"] = 2
        assert run.config["lr"] == 1 and run.config.a == 1
        assert isinstance(run.id, str) and run.url is None
    airon.log({"x": 2})
    airon.finish()
    assert threading.active_count() == threads_before
    assert server.requests == []
    assert not (tmp_path / "airon").exists()
    assert airon.run is None


def test_disabled_context_manager_still_reraises(monkeypatch):
    monkeypatch.setenv("AIRON_MODE", "disabled")
    with pytest.raises(KeyError), airon.init():
        raise KeyError("x")


# ---- distributed ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"RANK": "1"},
        {"LOCAL_RANK": "3"},
        {"RANK": "8", "LOCAL_RANK": "0"},
        {"RANK": "0", "LOCAL_RANK": "1"},
    ],
)
def test_other_ranks_get_a_no_op_run(server, start, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with start() as run:
        run.log({"x": 1})
        run.config["a"] = 1
        run.finish()
    assert server.requests == [] and type(run).__name__ == "NoopRun"


@pytest.mark.parametrize(
    "env", [{}, {"RANK": "0"}, {"RANK": "0", "LOCAL_RANK": "0"}, {"RANK": "oops"}]
)
def test_rank_zero_logs(server, start, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run = start()
    run.log({"x": 1})
    run.finish()
    assert len(server.event_list()) == 1


def test_rank_zero_only_false_lets_every_rank_log(server, start, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    run = start(rank_zero_only=False)
    run.log({"x": 1})
    run.finish()
    assert len(server.event_list()) == 1


# ---- identity -------------------------------------------------------------------------------


def test_project_and_run_id_from_environment(server, start, monkeypatch):
    monkeypatch.setenv("AIRON_PROJECT", "from-env")
    monkeypatch.setenv("AIRON_RUN_ID", "env-run-1")
    run = start(project=None)
    run.finish()
    assert run.id == "env-run-1" and server.run("env-run-1").creates[0]["project"] == "from-env"


def test_project_defaults_to_the_working_directory_name(server, start, tmp_path, monkeypatch):
    work = tmp_path / "my-model"
    work.mkdir()
    monkeypatch.chdir(work)
    run = start(project=None)
    run.finish()
    assert server.only_run().creates[0]["project"] == "my-model"


def test_run_ids_are_unique_uuids(server, start):
    ids = set()
    for _ in range(3):
        run = start()
        ids.add(run.id)
        run.finish()
    assert len(ids) == 3


# ---- resuming -------------------------------------------------------------------------------


def test_resume_continues_seq_and_step_after_the_servers_last_values(server, start):
    first = start(run_id="resume-me", name="first")
    for i in range(10):
        first.log({"x": i})
    first.finish()
    assert server.run("resume-me").last_seq == 10

    second = start(run_id="resume-me", resume="allow")
    second.log({"x": 99})
    second.finish()
    state = server.run("resume-me")
    assert state.creates[1]["resume"] == "allow"
    last = state.events[max(state.events)]
    assert last["seq"] == 11 and last["step"] == 10 and last["metrics"] == {"x": 99}
    assert sorted(state.events) == list(range(1, 12))
    assert state.status == "finished"


def test_resume_true_means_allow(server, start):
    first = start(run_id="r")
    first.finish()
    second = start(run_id="r", resume=True)
    second.finish()
    assert server.run("r").creates[1]["resume"] == "allow"


def test_explicit_steps_win_when_resuming(server, start):
    first = start(run_id="r2")
    first.log({"x": 1}, step=500)
    first.finish()
    second = start(run_id="r2", resume="allow")
    second.log({"x": 2}, step=700)
    second.log({"x": 3})
    second.finish()
    assert [e["step"] for e in server.event_list("r2")][-2:] == [700, 701]


def test_finished_run_without_resume_is_a_conflict_that_does_not_raise(
    server, start, capsys, spool_root
):
    first = start(run_id="done")
    first.finish()
    second = start(run_id="done")
    second.log({"x": 1})
    second.finish()
    err = capsys.readouterr().err
    assert "already exists and is over" in err and "resume='allow'" in err
    assert server.run("done").events == {}
    assert any(spool_root.iterdir())  # kept, and `airon sync` can reopen the run


def test_resume_must_for_an_unknown_run_does_not_raise(server, start, capsys):
    run = start(run_id="never-existed", resume="must")
    run.log({"x": 1})
    run.finish()
    assert "does not exist on the server" in capsys.readouterr().err
    assert not server.runs


def test_unknown_resume_value_warns(server, start, capsys):
    run = start(resume="maybe")
    run.finish()
    assert "unknown resume value" in capsys.readouterr().err
    assert server.only_run().creates[0]["resume"] == "never"


# ---- secrets ----------------------------------------------------------------------------------


def test_the_key_never_appears_in_output_or_requests(server, start, capsys, monkeypatch):
    monkeypatch.delenv("AIRON_QUIET")
    secret = "super-secret-key-0042"
    server.keys.add(secret)
    run = start(api_key=secret)
    run.log({"x": 1})
    run.finish()
    server.inject("/events", 500, times=100)
    bad = start(api_key=secret, retry_attempts=1)
    bad.log({"x": 1})
    bad.finish()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    for request in server.requests:
        assert secret not in request.path and secret not in json.dumps(request.body)
    assert os.environ.get("AIRON_API_KEY") is None
