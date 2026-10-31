from __future__ import annotations

import time


def test_events_are_sent_in_order_and_in_batches_of_at_most_batch_size(server, start):
    run = start(batch_size=50, flush_interval=5.0)
    for i in range(240):
        run.log({"x": i})
    run.finish()
    batches = [r.body["events"] for r in server.calls("POST", "/events")]
    assert all(len(b) <= 50 for b in batches)
    assert sum(len(b) for b in batches) == 240
    seqs = [e["seq"] for b in batches for e in b]
    assert seqs == sorted(seqs) == list(range(1, 241))
    assert [e["metrics"]["x"] for e in server.event_list()] == list(range(240))


def test_a_full_batch_is_sent_before_the_flush_interval(server, start):
    run = start(batch_size=20, flush_interval=30.0)
    server.wait_for(lambda: server.runs)
    for i in range(20):
        run.log({"x": i})
    assert server.wait_for(lambda: server.calls("POST", "/events"), timeout=3)
    run.finish()


def test_events_flush_on_the_interval_without_filling_a_batch(server, start):
    run = start(flush_interval=0.1)
    run.log({"x": 1})
    assert server.wait_for(lambda: server.calls("POST", "/events"), timeout=3)
    assert server.event_list()[0]["metrics"] == {"x": 1}
    run.finish()


def test_large_bodies_are_gzipped_and_small_ones_are_not(server, start):
    run = start(batch_size=200)
    run.log({"x": 1})
    time.sleep(0.3)
    for i in range(150):
        run.log({f"layer{i}/grad_norm": 0.123456789 * i for i in range(10)})
    run.finish()
    events = server.calls("POST", "/events")
    small = [r for r in events if r.raw_size <= 1024]
    big = [r for r in events if r.raw_size > 1024]
    assert small and big
    assert all(not r.gzipped for r in small)
    assert all(r.gzipped for r in big)
    assert all(r.headers["Content-Encoding"] == "gzip" for r in big)
    assert len(server.event_list()) == 151  # the server decoded every gzipped batch


def test_idle_runs_send_heartbeats(server, start):
    run = start(heartbeat_interval=0.15, flush_interval=0.05)
    assert server.wait_for(
        lambda: [e for e in server.event_list() if set(e) == {"seq", "t"}], timeout=5
    )
    run.finish()
    beats = [e for e in server.event_list() if set(e) == {"seq", "t"}]
    assert beats and all(isinstance(e["t"], float) for e in beats)


def test_busy_runs_do_not_add_heartbeats(server, start):
    run = start(heartbeat_interval=1.0, flush_interval=0.05)
    end = time.time() + 1.5
    while time.time() < end:
        run.log({"x": 1})
        time.sleep(0.02)
    run.finish()
    assert not [e for e in server.event_list() if set(e) == {"seq", "t"}]


def test_oversized_batches_are_split_when_the_server_answers_413(server, start):
    server.max_body = 3000
    run = start(batch_size=200)
    for i in range(200):
        run.log({"metric": float(i)})
    run.finish()
    assert [e["metrics"]["metric"] for e in server.event_list()] == [float(i) for i in range(200)]
    assert any(r for r in server.calls("POST", "/events") if r.raw_size > 3000)  # first try failed
    assert server.only_run().status == "finished"
