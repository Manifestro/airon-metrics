from __future__ import annotations

import re
import time
import uuid

import airon


def test_run_is_created_logged_and_finished(server, start):
    run = start(
        name="baseline-v2",
        config={"lr": 3e-5, "model": {"name": "small"}},
        experiment="Baseline v2",
        tags=["a", "b"],
        notes="hello",
        total_steps=100,
        total_epochs=5,
    )
    assert run.name == "baseline-v2"
    assert str(uuid.UUID(run.id)) == run.id
    run.log({"train/loss": 0.5, "lr": 1e-4}, step=10, epoch=0.5)
    run.finish()

    state = server.only_run()
    create = state.creates[0]
    assert create["project"] == "proj"
    assert create["name"] == "baseline-v2"
    assert create["experiment"] == "Baseline v2"
    assert create["config"] == {"lr": 3e-5, "model": {"name": "small"}}
    assert create["tags"] == ["a", "b"]
    assert create["notes"] == "hello"
    assert create["total_steps"] == 100
    assert create["total_epochs"] == 5
    assert create["resume"] == "never"
    assert abs(create["started_at"] - time.time()) < 30
    assert "python" in create["env"] and "platform" in create["env"] and "host" in create["env"]

    (event,) = server.event_list()
    assert event["metrics"] == {"train/loss": 0.5, "lr": 1e-4}
    assert event["step"] == 10 and event["epoch"] == 0.5 and event["seq"] == 1
    assert event["progress"]["total_steps"] == 100
    assert state.status == "finished"
    assert state.finish["status"] == "finished"


def test_requests_carry_auth_protocol_and_user_agent(server, start):
    run = start()
    run.log({"x": 1})
    run.finish()
    for request in server.requests:
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["X-Airon-Protocol"] == "1"
        assert re.match(r"airon/\d+\.\d+\.\d+ python/3\.\d+", request.headers["User-Agent"])
        assert "test-key" not in request.path


def test_name_is_generated_locally_when_missing(server, start):
    run = start()
    assert re.fullmatch(r"[a-z]+-[a-z]+-\d{1,2}", run.name)
    run.finish()
    assert server.only_run().creates[0]["name"] == run.name


def test_run_url_comes_from_the_server_and_is_known_immediately(server, start):
    run = start()
    assert run.url == f"{server.url}/runs/{run.id}"
    run.finish()


def test_step_counter_starts_at_zero_and_follows_explicit_steps(server, start):
    run = start()
    run.log({"x": 1})
    run.log({"x": 2})
    run.log({"x": 3}, step=50)
    run.log({"x": 4})
    run.finish()
    assert [e["step"] for e in server.event_list()] == [0, 1, 50, 51]


def test_seq_is_contiguous_from_one(server, start):
    run = start()
    for i in range(30):
        run.log({"x": i})
    run.finish()
    assert [e["seq"] for e in server.event_list()] == list(range(1, 31))


def test_progress_rates_use_step_deltas_not_call_counts(server, start):
    run = start(total_steps=1000)
    run.log({"x": 1}, step=0, samples=8, tokens=1024)
    time.sleep(0.2)
    run.log({"x": 1}, step=100, samples=8, tokens=1024)  # logging every 100 steps
    run.finish()
    first, second = (e["progress"] for e in server.event_list())
    assert "steps_per_sec" not in first
    # 100 steps in about 0.2 s: roughly 500 steps/s; a per-call rate would be about 5/s
    assert 100 < second["steps_per_sec"] < 1000
    assert (
        second["samples_per_sec"] == round(second["steps_per_sec"] * 8, 4)
        or abs(second["samples_per_sec"] - second["steps_per_sec"] * 8) < 0.01
    )
    assert second["tokens_per_sec"] > second["samples_per_sec"]
    assert second["step"] == 100 and second["total_steps"] == 1000


def test_rate_is_smoothed_over_several_logs(server, start):
    run = start()
    run.log({"x": 1}, step=0)
    for step in range(1, 6):
        time.sleep(0.05)
        run.log({"x": 1}, step=step)
    time.sleep(0.4)  # a long pause, like an evaluation
    run.log({"x": 1}, step=6)
    run.finish()
    rates = [e["progress"]["steps_per_sec"] for e in server.event_list()[1:]]
    assert rates[-1] < rates[-2]  # the pause pulls the rate down
    assert rates[-1] > 1.0  # but smoothing keeps it from collapsing to one reading


def test_no_rate_when_the_step_does_not_advance(server, start):
    run = start()
    run.log({"x": 1}, step=5)
    run.log({"eval/loss": 1}, step=5)
    run.finish()
    assert all("steps_per_sec" not in e["progress"] for e in server.event_list())


def test_progress_call_sends_totals_without_metrics(server, start):
    run = start()
    run.log({"x": 1}, step=3, epoch=0.1)
    run.progress(total_steps=500, epoch=1.0, total_epochs=4)
    run.finish()
    progress = server.event_list()[-1]["progress"]
    assert progress["step"] == 3
    assert progress["total_steps"] == 500
    assert progress["epoch"] == 1.0 and progress["total_epochs"] == 4
    assert "metrics" not in server.event_list()[-1]


def test_config_and_summary_assignments_send_events(server, start):
    run = start(config={"lr": 1})
    run.config["batch_size"] = 32
    run.config.update({"seed": 7})
    run.config.warmup = 100
    run.summary["eval/loss.best"] = 0.43
    assert run.config["batch_size"] == 32 and run.config.seed == 7
    assert dict(run.summary) == {"eval/loss.best": 0.43}
    run.finish()
    events = server.event_list()
    assert [e["config"] for e in events if "config" in e] == [
        {"batch_size": 32},
        {"seed": 7},
        {"warmup": 100},
    ]
    assert [e["summary"] for e in events if "summary" in e] == [{"eval/loss.best": 0.43}]
    assert server.only_run().finish["summary"] == {"eval/loss.best": 0.43}


def test_secret_looking_config_values_are_masked(server, start):
    run = start(config={"hf_token": "hf_abcdefgh", "lr": 1})
    run.config["wandb_api_key"] = "zzz"
    run.finish()
    sent = str([server.only_run().creates, server.event_list()])
    assert "hf_abcdefgh" not in sent and "zzz" not in sent


def test_checkpoint_is_a_pointer_only(server, start):
    run = start()
    run.log({"x": 1}, step=12000)
    run.checkpoint("hf://org/repo@abc", label="best", **{"eval/loss": 0.4})
    run.checkpoint("s3://bucket/ckpt-3", step=3)
    run.finish()
    first, second = [e["checkpoint"] for e in server.event_list() if "checkpoint" in e]
    assert first == {
        "uri": "hf://org/repo@abc",
        "step": 12000,
        "label": "best",
        "metrics": {"eval/loss": 0.4},
    }
    assert second == {"uri": "s3://bucket/ckpt-3", "step": 3}


def test_module_level_shortcuts_follow_the_current_run(server, start):
    assert airon.run is None
    run = start()
    assert airon.run is run
    airon.log({"x": 1})
    airon.finish()
    assert airon.run is None
    assert server.only_run().status == "finished"
    assert len(server.event_list()) == 1


def test_log_without_a_run_warns_and_does_nothing(capsys):
    airon.log({"x": 1})
    assert "no active run" in capsys.readouterr().err


def test_calls_after_finish_are_ignored(server, start):
    run = start()
    run.log({"x": 1})
    run.finish()
    run.log({"x": 2})
    run.finish()
    run.checkpoint("s3://x")
    assert len(server.event_list()) == 1
    assert server.only_run().finish_count == 1


def test_starting_a_second_run_finishes_the_first(server, start):
    first = start(name="one")
    second = start(name="two")
    second.finish()
    assert first.id != second.id
    assert {s.status for s in server.runs.values()} == {"finished"}


def test_server_warnings_are_shown_once(server, start, capsys):
    server.warnings = ["experiment 'Nope' not found"]
    run = start(experiment="Nope")
    run.log({"x": 1})
    run.finish()
    assert capsys.readouterr().err.count("experiment 'Nope' not found") == 1


def test_terminal_lines_and_quiet(server, start, monkeypatch, capsys):
    monkeypatch.delenv("AIRON_QUIET")
    run = start(name="baseline-v2")
    run.finish()
    err = capsys.readouterr().err
    assert f"airon: tracking run baseline-v2 -> {server.url}/runs/{run.id}" in err
    assert "airon: run baseline-v2 finished" in err
    assert "test-key" not in err

    monkeypatch.setenv("AIRON_QUIET", "1")
    run = start(name="quiet")
    run.finish()
    assert capsys.readouterr().err == ""
