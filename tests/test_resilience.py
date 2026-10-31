"""The SDK must never raise into, or hold up, the training loop."""

from __future__ import annotations

import socket
import threading
import time
from collections import deque

import airon
from airon._sender import Sender, Tuning
from airon._spool import Spool


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return time.perf_counter() - started, result


def log_many(run, count=2000):
    started = time.perf_counter()
    for i in range(count):
        run.log({"train/loss": 1.0 / (i + 1), "lr": 1e-4}, samples=8)
    return time.perf_counter() - started


def test_server_refusing_connections(server, start, spool_root):
    server.stop()
    elapsed, run = timed(start)
    assert elapsed < 0.5
    assert log_many(run) < 1.0
    elapsed, _ = timed(run.finish)
    assert elapsed < 6.0  # bounded by the shutdown timeout, usually far less
    assert any(spool_root.iterdir())


def test_very_slow_server_does_not_slow_logging_and_exit_is_capped(server, start, spool_root):
    server.latency = 4.0
    elapsed, run = timed(start, request_timeout=5.0, shutdown_timeout=1.0)
    assert elapsed < 0.5
    assert log_many(run) < 1.0
    elapsed, _ = timed(run.finish)
    assert elapsed < 3.0  # one second of patience, plus the time to write to disk
    events = [p for p in spool_root.rglob("events-*.jsonl")]
    assert events  # what could not be delivered in time went to disk


def test_requests_that_time_out_are_retried_not_raised(server, start):
    server.latency = 0.5
    run = start(request_timeout=0.2, retry_attempts=2, shutdown_timeout=1.0)
    run.log({"x": 1})
    elapsed, _ = timed(run.finish)
    assert elapsed < 4.0


def test_garbage_replies(server, start):
    server.garbage = True
    run = start(retry_attempts=2)
    assert log_many(run, 200) < 1.0
    elapsed, _ = timed(run.finish)
    assert elapsed < 6.0


def test_dns_failure(start, monkeypatch, spool_root):
    def no_such_host(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_host)
    run = start(url="https://airon.example.com")
    assert log_many(run, 200) < 1.0
    elapsed, _ = timed(run.finish)
    assert elapsed < 6.0
    assert any(spool_root.iterdir())


def test_log_from_many_threads(server, start):
    run = start(batch_size=100)
    threads = 16
    per_thread = 500
    errors: list[BaseException] = []

    def work():
        try:
            for i in range(per_thread):
                run.log({"x": i})
        except BaseException as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    pool = [threading.Thread(target=work) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(30)
    run.finish()
    assert not errors
    events = server.event_list()
    assert len(events) == threads * per_thread
    assert [e["seq"] for e in events] == list(range(1, threads * per_thread + 1))
    assert sorted(e["step"] for e in events) == list(range(threads * per_thread))  # no step reused


def test_log_from_many_threads_while_the_server_is_down(server, start, spool_root):
    server.stop()
    run = start()

    def work():
        for i in range(300):
            run.log({"x": i})

    pool = [threading.Thread(target=work) for _ in range(8)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(30)
        assert not thread.is_alive()
    run.finish()


def test_bad_arguments_never_raise(server, start):
    run = start(config=object(), total_steps="many", tags=5, resume="sometimes", mode="warp")
    run.log("not a dict")
    run.log({"x": object(), "y": float("nan"), 3: 4})
    run.log({"x": 1}, step="soon", epoch="later", samples="lots")
    run.progress(step=object(), total_steps="?")
    run.checkpoint(None, step="x")
    run.config["ok"] = object()
    run.summary.update({"a": {1, 2}})
    run.finish(status="exploded", exit_code="x")


def test_weird_init_arguments_give_a_working_noop_run(capsys):
    run = airon.init(url=7, api_key=[], flush_interval="soon")
    run.log({"x": 1})
    run.finish()
    assert "could not start tracking" in capsys.readouterr().err


def test_exit_flush_is_complete_when_the_server_is_healthy(server, start):
    run = start(flush_interval=30.0)  # nothing would be sent before the end without the exit flush
    for i in range(100):
        run.log({"x": i})
    run.finish()
    assert len(server.event_list()) == 100
    assert server.only_run().status == "finished"


def make_sender(tmp_path, max_queue):
    return Sender(
        "r",
        Spool(tmp_path / "r"),
        None,
        {"run_id": "r"},
        Tuning(max_queue=max_queue),
        dict,
        lambda data: None,
        list,
    )


def test_queue_overflow_drops_system_samples_first_then_metrics(tmp_path, capsys):
    sender = make_sender(tmp_path, max_queue=100)
    for i in range(50):
        sender.enqueue({"t": float(i), "system": {"cpu/util_pct": 1}})
    for i in range(60):
        sender.enqueue({"t": 100.0 + i, "metrics": {"x": i}})
    kept = list(sender._q)
    assert len(kept) <= 100
    assert sum("metrics" in e for e in kept) == 60  # no metric lost yet
    systems = [e for e in kept if "system" in e]
    assert 0 < len(systems) < 50 and systems[0]["t"] > 0  # the oldest samples went first
    assert capsys.readouterr().err.count("event queue is full") == 1

    for i in range(200):
        sender.enqueue({"t": 1000.0 + i, "metrics": {"x": 1000 + i}})
    kept = list(sender._q)
    assert len(kept) <= 100
    assert not [e for e in kept if "system" in e]  # every sample is gone before a metric is
    newest = [e["metrics"]["x"] for e in kept if "metrics" in e]
    assert newest[-1] == 1199 and newest == sorted(newest)  # the newest metrics survive
    assert sender.stats.dropped == 310 - len(kept)
    assert capsys.readouterr().err == ""  # still only one warning


def test_overflow_protects_config_and_checkpoints_until_last(tmp_path):
    sender = make_sender(tmp_path, max_queue=100)
    sender.enqueue({"t": 0.0, "config": {"a": 1}})
    sender.enqueue({"t": 0.0, "checkpoint": {"uri": "s3://x"}})
    for i in range(300):
        sender.enqueue({"t": float(i), "metrics": {"x": i}})
    kinds = [next(k for k in ("config", "checkpoint", "metrics") if k in e) for e in sender._q]
    assert "config" in kinds and "checkpoint" in kinds


def test_logging_stays_bounded_while_the_sender_is_stuck(server, start, spool_root):
    server.latency = 3.0
    run = start(max_queue=500, request_timeout=5.0, shutdown_timeout=0.5, batch_size=50)
    for i in range(20000):
        run.log({"x": i})
    sender_queue = run._sender._q
    assert isinstance(sender_queue, deque) and len(sender_queue) <= 500
    run.finish()
