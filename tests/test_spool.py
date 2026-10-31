from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import airon
from airon import _spool, cli


def run_dir(spool_root: Path) -> Path:
    (directory,) = [d for d in spool_root.iterdir() if d.is_dir()]
    return directory


def spooled_events(directory: Path) -> list[dict]:
    lines = []
    for segment in sorted(directory.glob("events-*.jsonl")):
        lines += [json.loads(line) for line in segment.read_text().splitlines()]
    return lines


def configure(monkeypatch, server, key="test-key"):
    monkeypatch.setenv("AIRON_URL", server.url)
    monkeypatch.setenv("AIRON_API_KEY", key)


def test_offline_mode_never_touches_the_network(server, start, spool_root):
    run = start(mode="offline", name="off")
    for i in range(10):
        run.log({"train/loss": 1 / (i + 1)}, step=i)
    run.finish()
    assert server.requests == []
    directory = run_dir(spool_root)
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["create"]["name"] == "off" and meta["create"]["project"] == "proj"
    assert "env" in meta["create"]
    assert [e["seq"] for e in spooled_events(directory)] == list(range(1, 11))
    assert json.loads((directory / "finish.json").read_text())["status"] == "finished"


def test_spool_files_are_json_lines_and_hold_no_key(start, spool_root):
    run = start(mode="offline", api_key="not-a-real-secret")
    run.log({"x": 1})
    run.finish()
    for path in run_dir(spool_root).iterdir():
        assert "not-a-real-secret" not in path.read_text()
    segment = next(run_dir(spool_root).glob("events-*.jsonl"))
    assert all(json.loads(line) for line in segment.read_text().splitlines())


def test_sync_uploads_in_order_and_cleans_up(server, start, spool_root, monkeypatch, capsys):
    run = start(mode="offline", name="off", batch_size=500)
    for i in range(1200):
        run.log({"x": i}, step=i)
    run.config["late"] = True
    run.finish()
    directory = run_dir(spool_root)
    sizes = [len(p.read_text().splitlines()) for p in sorted(directory.glob("events-*.jsonl"))]
    assert len(sizes) >= 3 and max(sizes) <= 500  # segments are bounded

    configure(monkeypatch, server)
    assert cli.main(["sync"]) == 0
    out = capsys.readouterr().out
    assert "off" in out and "1,201 events" in out
    state = server.only_run()
    assert state.creates[0]["resume"] == "allow"
    events = server.event_list()
    assert [e["seq"] for e in events] == list(range(1, 1202))
    assert [e["metrics"]["x"] for e in events if "metrics" in e] == list(range(1200))
    assert state.status == "finished"
    assert not directory.exists()
    assert cli.main(["sync"]) == 0  # nothing left to do


def test_sync_without_configuration_explains(start, spool_root, capsys):
    run = start(mode="offline")
    run.finish()
    assert cli.main(["sync"]) == 2
    assert "AIRON_URL" in capsys.readouterr().err
    assert run_dir(spool_root).exists()  # nothing was thrown away


def test_sync_reports_failure_and_keeps_the_spool(server, start, spool_root, monkeypatch, capsys):
    run = start(mode="offline")
    run.log({"x": 1})
    run.finish()
    configure(monkeypatch, server)
    server.inject("=POST /v1/runs", 500, times=4)
    monkeypatch.setattr("airon._sync.time.sleep", lambda s: None)
    assert cli.main(["sync"]) == 1
    assert "failed" in capsys.readouterr().out
    assert run_dir(spool_root).exists()
    assert cli.main(["sync"]) == 0  # the server recovered
    assert server.only_run().status == "finished"


def test_sync_with_a_bad_key_fails_clearly_and_keeps_the_spool(
    server, start, spool_root, monkeypatch, capsys
):
    run = start(mode="offline")
    run.finish()
    configure(monkeypatch, server, key="wrong-wrong-wrong")
    assert cli.main(["sync"]) == 1
    captured = capsys.readouterr()
    assert (
        "rejected the key" in captured.err and "wrong-wrong-wrong" not in captured.err + captured.out
    )
    assert run_dir(spool_root).exists()


def test_sync_skips_runs_that_are_still_alive_unless_forced(
    server, start, spool_root, monkeypatch, capsys
):
    run = start(mode="offline")
    run.log({"x": 1})
    run.finish()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        meta_path = run_dir(spool_root) / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["pid"] = sleeper.pid
        meta_path.write_text(json.dumps(meta))
        configure(monkeypatch, server)
        assert cli.main(["sync"]) == 0
        assert "skipped" in capsys.readouterr().out and not server.runs
        assert cli.main(["sync", "--force"]) == 0
        assert server.only_run().status == "finished"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_torn_last_line_after_a_kill_is_ignored(server, tmp_path, monkeypatch):
    spool = _spool.Spool(tmp_path / "r1")
    assert spool.append([{"seq": 1, "t": 1.0}, {"seq": 2, "t": 2.0}], {"create": {}})
    segment = spool.segments()[0]
    with open(segment, "a") as handle:
        handle.write('{"seq": 3, "t"')
    assert [e["seq"] for e in spool.read_segment(segment)] == [1, 2]
    assert spool.max_seq() == 2


def test_a_restarted_process_continues_the_numbering_on_disk(start, spool_root):
    first = start(mode="offline", run_id="same-run")
    first.log({"x": 1})
    first.log({"x": 2})
    first.finish()
    second = start(mode="offline", run_id="same-run")
    second.log({"x": 3})
    second.finish()
    seqs = [e["seq"] for e in spooled_events(run_dir(spool_root))]
    assert seqs == [1, 2, 3]


def test_unsafe_run_ids_cannot_escape_the_spool_directory(start, spool_root):
    run = start(mode="offline", run_id="../../evil/../x")
    run.finish()
    (directory,) = spool_root.iterdir()
    assert directory.parent == spool_root
    assert "/" not in directory.name and directory.name not in ("", ".", "..")
    assert _spool.directory_name("..") == "run"


def test_events_are_redelivered_automatically_when_the_server_returns(server, start, spool_root):
    run = start()
    run.log({"x": 0})
    assert server.wait_for(lambda: server.event_list())
    server.stop()
    for i in range(1, 30):
        run.log({"x": i})
        time.sleep(0.02)
    time.sleep(0.5)  # long enough for retries to give up and spool
    assert spool_root.exists() and any(spool_root.iterdir())
    server.restart()
    assert server.wait_for(lambda: len(server.event_list()) == 30, timeout=10)
    for i in range(30, 40):
        run.log({"x": i})
    run.finish()
    events = server.event_list()
    assert [e["seq"] for e in events] == list(range(1, 41))  # complete, in order, no repeats
    assert [e["metrics"]["x"] for e in events] == list(range(40))
    assert server.only_run().status == "finished"
    assert not any(spool_root.iterdir())  # the spool cleaned up after itself


def test_server_down_for_the_whole_run_leaves_everything_for_sync(
    server, start, spool_root, monkeypatch
):
    server.stop()
    run = start(name="down")
    for i in range(25):
        run.log({"x": i}, step=i)
    run.finish()
    directory = run_dir(spool_root)
    assert len(spooled_events(directory)) == 25
    server.start()  # same port
    configure(monkeypatch, server)
    assert cli.main(["sync"]) == 0
    assert [e["metrics"]["x"] for e in server.event_list()] == list(range(25))
    assert server.only_run().status == "finished"


def test_unwritable_spool_location_is_survivable(start, tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("AIRON_DIR", str(blocker / "sub"))
    run = start(mode="offline")
    run.log({"x": 1})
    run.finish()
    err = capsys.readouterr().err
    assert err.count("cannot write the offline spool") == 1


def test_full_disk_is_survivable(start, monkeypatch, capsys):
    def full(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_spool.Spool, "_write_chunk", full)
    run = start(mode="offline")
    for i in range(50):
        run.log({"x": i})
    run.finish()
    assert capsys.readouterr().err.count("cannot write the offline spool") == 1


@pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a non-root POSIX user",
)
def test_read_only_spool_directory_is_survivable(start, tmp_path, monkeypatch, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setenv("AIRON_DIR", str(locked))
    try:
        run = start(mode="offline")
        run.log({"x": 1})
        run.finish()
    finally:
        locked.chmod(0o700)
    assert "cannot write the offline spool" in capsys.readouterr().err


def test_airon_with_no_server_configured_explains_and_goes_offline(spool_root, capsys):
    run = airon.init(project="p", console=False, system_metrics=False)
    run.log({"x": 1})
    run.finish()
    err = capsys.readouterr().err
    assert run.mode == "offline"
    for word in ("AIRON_URL", "AIRON_API_KEY", "airon login"):
        assert word in err
    assert err.count("no usable server settings") == 1
    assert spooled_events(run_dir(spool_root))
