"""`log()` runs inside the training loop, so its cost is part of the contract."""

from __future__ import annotations

import statistics
import time

import airon


def measure(run, calls=20000):
    samples = []
    for i in range(calls):
        started = time.perf_counter_ns()
        run.log({"train/loss": 0.5, "train/acc": 0.9, "lr": 1e-4}, step=i, samples=32, tokens=4096)
        samples.append(time.perf_counter_ns() - started)
    return statistics.median(samples) / 1000, statistics.fmean(samples) / 1000


def test_log_costs_tens_of_microseconds_at_most(server, start):
    run = start(batch_size=200)
    median, mean = measure(run)
    run.finish()
    print(f"\nlog() against a live server: median {median:.1f} us, mean {mean:.1f} us")
    assert median < 50
    assert mean < 150
    assert len(server.event_list()) == 20000


def test_log_stays_cheap_while_the_server_is_down(server, start):
    server.stop()
    run = start()
    median, mean = measure(run)
    run.finish()
    print(f"\nlog() with the server down: median {median:.1f} us, mean {mean:.1f} us")
    assert median < 50 and mean < 150


def test_log_stays_cheap_when_the_queue_overflows(start, server):
    server.latency = 5.0
    run = start(max_queue=2000, request_timeout=10.0, shutdown_timeout=0.2)
    median, mean = measure(run)
    run.finish()
    print(f"\nlog() with a full queue: median {median:.1f} us, mean {mean:.1f} us")
    assert median < 50 and mean < 300


def test_disabled_log_is_nearly_free(monkeypatch):
    monkeypatch.setenv("AIRON_MODE", "disabled")
    run = airon.init()
    median, _ = measure(run, 5000)
    print(f"\nlog() when disabled: median {median:.2f} us")
    assert median < 5
