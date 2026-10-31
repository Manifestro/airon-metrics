"""The retry and drop rules of the protocol, status code by status code."""

from __future__ import annotations

import json
import time

import pytest

from airon import _log


def event_requests(server):
    return server.calls("POST", "/events")


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried_until_delivered(server, start, status):
    server.inject("/events", status, times=2)
    run = start()
    run.log({"x": 1})
    run.finish()
    assert len(event_requests(server)) == 3
    assert [e["metrics"] for e in server.event_list()] == [{"x": 1}]


@pytest.mark.parametrize("status", [400, 404, 405, 409, 410, 415, 422])
def test_other_4xx_drop_the_batch_and_warn_without_retrying(server, start, capsys, status):
    run = start(flush_interval=0.05)
    run.log({"x": 1})
    server.wait_for(lambda: server.event_list())  # let the first batch through
    server.inject("/events", status)
    run.log({"x": 2})
    time.sleep(0.4)
    run.log({"x": 3})
    run.finish()
    assert [e["metrics"]["x"] for e in server.event_list()] == [1, 3]  # x=2 was dropped
    assert len(event_requests(server)) == 3  # no retry of the refused batch
    assert capsys.readouterr().err.count(f"HTTP {status}") == 1  # warned once
    assert server.only_run().status == "finished"  # the run carries on


def test_retry_after_is_honoured(server, start):
    server.inject("/events", 429, retry_after=1)
    run = start()
    run.log({"x": 1})
    run.finish()
    first, second = event_requests(server)[:2]
    assert second.at - first.at >= 0.95
    assert len(server.event_list()) == 1


def test_backoff_grows_between_attempts(server, start):
    server.inject("/events", 503, times=3)
    run = start(retry_min_delay=0.1, retry_max_delay=5.0, retry_attempts=5)
    run.log({"x": 1})
    run.finish()
    times = [r.at for r in event_requests(server)]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(gaps) == 3
    assert gaps[0] >= 0.04 and gaps[1] >= 0.09 and gaps[2] >= 0.19  # 0.1, 0.2, 0.4 with jitter


def test_a_batch_whose_reply_was_lost_is_resent_and_deduplicated(server, start):
    server.lose_replies = 1  # the server stores the batch, then the connection dies
    run = start()
    for i in range(5):
        run.log({"x": i})
    run.finish()
    assert len(event_requests(server)) == 2
    assert [e["seq"] for e in server.event_list()] == [1, 2, 3, 4, 5]  # no duplicates stored
    assert server.only_run().status == "finished"


def test_create_is_retried_in_the_background_and_events_follow(server, start):
    server.inject("=POST /v1/runs", 503, times=2)
    run = start()
    run.log({"x": 1})
    assert server.wait_for(lambda: server.runs, timeout=20)  # created on a later attempt
    run.log({"x": 2})
    run.finish()
    assert len([r for r in server.requests if r.path == "/v1/runs"]) == 3
    assert [e["metrics"]["x"] for e in server.event_list()] == [1, 2]
    assert server.only_run().status == "finished"


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_stop_sending_and_keep_everything_on_disk(
    server, start, spool_root, capsys, status
):
    server.inject("/events", status, times=50)
    run = start(flush_interval=0.05)
    run.log({"x": 1})
    server.wait_for(lambda: event_requests(server))
    for i in range(2, 5):
        run.log({"x": i})
        time.sleep(0.1)
    run.finish()
    assert len(event_requests(server)) == 1  # sending stopped after the first refusal
    err = capsys.readouterr().err
    assert err.count("rejected the API key") == 1
    assert "test-key" not in err
    (directory,) = spool_root.iterdir()
    lines = [
        json.loads(line)
        for f in sorted(directory.glob("events-*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    assert [e["metrics"]["x"] for e in lines] == [1, 2, 3, 4]
    assert [e["seq"] for e in lines] == [1, 2, 3, 4]
    assert json.loads((directory / "finish.json").read_text())["status"] == "finished"


def test_wrong_key_at_create_stops_everything(server, start, spool_root, capsys):
    run = start(api_key="wrong-wrong-wrong")
    for i in range(5):
        run.log({"x": i})
    run.finish()
    assert not server.runs
    assert len(server.requests) == 1
    assert "wrong-wrong-wrong" not in capsys.readouterr().err
    (directory,) = spool_root.iterdir()
    assert (directory / "meta.json").exists()


def test_rejected_events_are_reported_once_and_not_resent(server, start, capsys):
    server.reject_names = {"bad"}
    run = start()
    run.log({"good": 1})
    run.log({"bad": 1})
    run.log({"also_good": 2})
    run.finish()
    keys = [next(iter(e["metrics"])) for e in server.event_list()]
    assert keys == ["good", "also_good"]
    assert len(event_requests(server)) == 1  # the refusal is final: nothing is resent
    err = capsys.readouterr().err
    assert err.count("refused 1 event") == 1 and "metrics.bad is not a number" in err


def test_server_with_garbage_replies_is_treated_as_down(server, start, spool_root):
    server.garbage = True
    run = start(retry_attempts=2)
    run.log({"x": 1})
    run.finish()
    assert not server.runs
    assert any(spool_root.iterdir())


def test_one_warning_per_kind_of_problem(server, start, capsys):
    server.inject("/events", 500, times=100)
    run = start(retry_attempts=1)
    for i in range(20):
        run.log({"x": i})
        time.sleep(0.01)
    run.finish()
    err = capsys.readouterr().err
    assert err.count("cannot reach the Airon server") == 1


def test_warn_once_helper(capsys):
    _log.reset()
    for _ in range(3):
        _log.warn_once("kind", "problem")
    _log.warn_once("other", "different")
    assert capsys.readouterr().err.count("airon: warning:") == 2
