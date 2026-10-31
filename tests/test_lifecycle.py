"""How a run ends: context manager, unhandled exceptions, normal exit, SIGTERM, fork."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

import airon

PRELUDE = """
import os, signal, sys, time
import airon
run = airon.init(project="p", name="child", console=False, system_metrics=False,
                 flush_interval=0.05, retry_min_delay=0.02, shutdown_timeout=5)
"""


def run_script(server, body, *args, timeout=60, prelude=True):
    env = dict(os.environ, AIRON_URL=server.url, AIRON_API_KEY="test-key", AIRON_QUIET="1")
    code = textwrap.dedent(PRELUDE) if prelude else "import os, signal, sys, time\nimport airon\n"
    code += textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---- context manager --------------------------------------------------------------------


def test_clean_block_finishes(server, start):
    with start() as run:
        run.log({"x": 1})
    assert server.only_run().status == "finished"


def test_exception_in_block_is_reported_with_traceback_and_reraised(server, start):
    def failing_step():
        raise ValueError("loss is nan")

    with pytest.raises(ValueError, match="loss is nan"), start() as run:
        run.log({"x": 1})
        failing_step()
    state = server.only_run()
    assert state.status == "failed"
    error = state.finish["error"]
    assert error["type"] == "ValueError" and error["message"] == "loss is nan"
    assert "failing_step" in error["traceback"] and "Traceback" in error["traceback"]
    assert state.finish["exit_code"] == 1
    assert len(server.event_list()) == 1  # events logged before the failure arrived


def test_exception_types_are_qualified_by_module(server, start):
    import json

    with pytest.raises(json.JSONDecodeError), start():
        json.loads("{")
    assert server.only_run().finish["error"]["type"] == "json.decoder.JSONDecodeError"


def test_keyboard_interrupt_is_killed_not_failed(server, start):
    with pytest.raises(KeyboardInterrupt), start():
        raise KeyboardInterrupt
    state = server.only_run()
    assert state.status == "killed" and state.finish["exit_code"] == 130


def test_sys_exit_inside_the_block(server, start):
    with pytest.raises(SystemExit), start():
        sys.exit(0)
    assert server.only_run().status == "finished"


def test_sys_exit_with_an_error_code_fails_the_run(server, start):
    with pytest.raises(SystemExit), start():
        sys.exit(2)
    state = server.only_run()
    assert state.status == "failed" and state.finish["exit_code"] == 2


def test_explicit_status_and_exit_code(server, start):
    run = start()
    run.finish(status="failed", exit_code=7)
    assert server.only_run().finish["status"] == "failed"
    assert server.only_run().finish["exit_code"] == 7


def test_finishing_twice_after_a_failure_keeps_the_first_status(server, start):
    with pytest.raises(RuntimeError), start() as run:
        raise RuntimeError("x")
    run.finish()
    assert server.only_run().status == "failed"
    assert server.only_run().finish_count == 1


# ---- processes -----------------------------------------------------------------------------


def test_unhandled_exception_in_a_script(server):
    done = run_script(
        server,
        """
        run.log({"x": 1})
        def inner():
            raise ValueError("kaboom")
        inner()
        """,
    )
    assert done.returncode == 1
    assert "Traceback" in done.stderr and "kaboom" in done.stderr  # the normal report still appears
    state = server.only_run()
    assert state.status == "failed"
    assert state.finish["error"]["type"] == "ValueError"
    assert "inner" in state.finish["error"]["traceback"]
    assert len(server.event_list()) == 1


def test_an_existing_excepthook_is_chained_not_replaced(server):
    done = run_script(
        server,
        """
        run.finish()
        sys.excepthook = lambda *a: print("custom hook ran", file=sys.stderr)
        import airon._hooks as hooks
        hooks._installed = False  # as if airon had not been used yet in this process
        run2 = airon.init(project="p", name="chained", console=False, system_metrics=False,
                          flush_interval=0.05)
        assert sys.excepthook is not hooks._previous_excepthook
        raise RuntimeError("x")
        """,
    )
    assert "custom hook ran" in done.stderr
    state = next(s for s in server.runs.values() if s.creates[0]["name"] == "chained")
    assert state.status == "failed"


def test_failure_is_still_seen_when_something_replaces_our_excepthook(server):
    done = run_script(
        server,
        """
        sys.excepthook = lambda *a: print("someone else's hook", file=sys.stderr)
        raise RuntimeError("late replacement")
        """,
    )
    assert "someone else's hook" in done.stderr
    state = server.only_run()
    assert state.status == "failed"
    assert state.finish["error"]["message"] == "late replacement"


def test_normal_exit_without_finish_still_flushes_and_finishes(server):
    done = run_script(
        server,
        """
        for i in range(50):
            run.log({"x": i})
        """,
    )
    assert done.returncode == 0
    assert len(server.event_list()) == 50
    assert server.only_run().status == "finished"


def test_unhandled_keyboard_interrupt_is_killed(server):
    done = run_script(server, "run.log({'x': 1})\nraise KeyboardInterrupt")
    assert done.returncode != 0
    assert server.only_run().status == "killed"


def test_sigterm_reports_killed_after_the_programs_own_handler_ran(server, tmp_path):
    marker = tmp_path / "handler-ran"
    done = run_script(
        server,
        """
        def mine(signum, frame):
            with open(sys.argv[1], "w") as handle:
                handle.write(str(time.time()))
            print("custom handler ran", flush=True)
            sys.exit(3)
        signal.signal(signal.SIGTERM, mine)
        run = airon.init(project="p", name="termed", console=False, system_metrics=False,
                         flush_interval=0.05, shutdown_timeout=5)
        run.log({"x": 1})
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        """,
        str(marker),
        prelude=False,
    )
    assert done.returncode == 3  # the program's own exit code is preserved
    assert "custom handler ran" in done.stdout
    state = server.only_run()
    assert state.status == "killed"
    assert state.finish["t"] >= float(marker.read_text())  # reported after the handler
    assert state.finish["error"]["type"] == "SIGTERM"
    assert len(state.events) == 1


def test_sigterm_without_a_handler_reports_killed_and_still_terminates(server):
    done = run_script(
        server,
        """
        run.log({"x": 1})
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        """,
    )
    assert done.returncode == -signal.SIGTERM
    state = server.only_run()
    assert state.status == "killed" and len(state.events) == 1


def test_sigterm_handled_gracefully_by_the_program_is_still_reported_as_killed(server):
    done = run_script(
        server,
        """
        stop = []
        signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
        run = airon.init(project="p", name="graceful", console=False, system_metrics=False,
                         flush_interval=0.05)
        run.log({"x": 1})
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop
        run.log({"x": 2})  # the program saves a last checkpoint, then ends normally
        run.finish()
        """,
        prelude=False,
    )
    assert done.returncode == 0, done.stderr
    state = server.only_run()
    assert state.status == "killed"
    assert len(state.events) == 2


def test_ignored_sigterm_is_left_alone(server):
    done = run_script(
        server,
        """
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        run = airon.init(project="p", console=False, system_metrics=False)
        os.kill(os.getpid(), signal.SIGTERM)
        print("still alive")
        """,
        prelude=False,
    )
    assert "still alive" in done.stdout and done.returncode == 0
    assert server.only_run().status == "finished"


def test_no_signal_handler_is_installed_off_the_main_thread(server):
    done = run_script(
        server,
        """
        import threading
        before = signal.getsignal(signal.SIGTERM)
        thread = threading.Thread(
            target=lambda: airon.init(project="p", console=False, system_metrics=False))
        thread.start()
        thread.join()
        print("unchanged" if signal.getsignal(signal.SIGTERM) is before else "CHANGED")
        """,
        prelude=False,
    )
    assert "unchanged" in done.stdout, done.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_does_not_finish_or_log_for_the_parent(server):
    done = run_script(
        server,
        """
        run.log({"parent": 1})
        pid = os.fork()
        if pid == 0:
            run.log({"child": 1})
            run.finish()
            sys.exit(0)
        os.waitpid(pid, 0)
        time.sleep(0.2)
        run.log({"parent": 2})
        run.finish()
        """,
    )
    assert done.returncode == 0, done.stderr
    state = server.only_run()
    assert state.finish_count == 1
    assert [list(e["metrics"]) for e in server.event_list()] == [["parent"], ["parent"]]


def test_a_killed_process_leaves_the_run_unfinished_for_the_server_to_mark_crashed(server):
    code = (
        textwrap.dedent(PRELUDE) + "run.log({'x': 1})\nprint('ready', flush=True)\ntime.sleep(60)"
    )
    env = dict(os.environ, AIRON_URL=server.url, AIRON_API_KEY="test-key", AIRON_QUIET="1")
    child = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert server.wait_for(lambda: server.event_list(), timeout=10)
        child.kill()  # SIGKILL: a preempted spot instance gets no chance to say goodbye
        child.wait(10)
    finally:
        child.kill()
        child.stdout.close()
    assert server.only_run().status == "running"
    assert airon.run is None
