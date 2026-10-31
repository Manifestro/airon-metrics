from __future__ import annotations

import builtins
import getpass
import socket
import stat
import subprocess
import sys

import pytest

import airon
from airon import _config, cli
from airon._transport import Result


def configure(monkeypatch, server, key="test-key"):
    monkeypatch.setenv("AIRON_URL", server.url)
    monkeypatch.setenv("AIRON_API_KEY", key)


def run_cli(*argv):
    return cli.main(list(argv))


# ---- version and usage ------------------------------------------------------------------------


def test_version(capsys):
    assert run_cli("version") == 0
    out = capsys.readouterr().out
    assert f"airon {airon.__version__}" in out and "protocol 1" in out


def test_python_dash_m_airon_works():
    done = subprocess.run(
        [sys.executable, "-m", "airon", "version"], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0 and "airon 0.1.0" in done.stdout


def test_no_command_prints_help_and_exits_2(capsys):
    assert run_cli() == 2
    assert "doctor" in capsys.readouterr().out


def test_unknown_command_exits_2(capsys):
    with pytest.raises(SystemExit) as stopped:
        run_cli("explode")
    assert stopped.value.code == 2


# ---- login / logout -----------------------------------------------------------------------------


def answer(monkeypatch, url, key):
    monkeypatch.setattr(builtins, "input", lambda prompt="": url)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": key)


def test_login_verifies_and_saves_with_private_permissions(server, monkeypatch, capsys):
    answer(monkeypatch, server.url + "/", "test-key")
    assert run_cli("login") == 0
    out = capsys.readouterr().out
    assert "Logged in" in out and "test-key" not in out
    path = _config.credentials_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    connection = _config.resolve_connection()
    assert connection.url == server.url and connection.api_key == "test-key"


def test_login_with_url_flag_only_asks_for_the_key(server, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("should not ask"))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "test-key")
    assert run_cli("login", "--url", server.url) == 0


def test_login_with_a_wrong_key_saves_nothing(server, monkeypatch, capsys):
    answer(monkeypatch, server.url, "wrong-wrong-wrong")
    assert run_cli("login") == 1
    captured = capsys.readouterr()
    assert "does not accept this key" in captured.err
    assert "wrong-wrong-wrong" not in captured.out + captured.err
    assert not _config.credentials_path().exists()


def test_login_with_an_unreachable_server_saves_nothing(server, monkeypatch, capsys):
    server.stop()
    answer(monkeypatch, server.url, "test-key")
    assert run_cli("login") == 1
    assert "Nothing accepts connections" in capsys.readouterr().err
    assert not _config.credentials_path().exists()


def test_login_rejects_a_malformed_address(monkeypatch, capsys):
    answer(monkeypatch, "ftp://nope", "test-key")
    assert run_cli("login") == 2
    assert "not a valid address" in capsys.readouterr().err


def test_login_can_be_cancelled(monkeypatch):
    def cancel(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", cancel)
    assert run_cli("login") == 2


def test_logout(server, capsys):
    _config.save_credentials(server.url, "test-key")
    assert run_cli("logout") == 0
    assert not _config.credentials_path().exists()
    assert run_cli("logout") == 0
    assert "No saved credentials" in capsys.readouterr().out


# ---- doctor ---------------------------------------------------------------------------------------


def test_doctor_success_writes_a_test_run_and_prints_its_link(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    assert run_cli("doctor") == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "Everything works" in out
    state = server.only_run()
    assert state.status == "finished" and state.creates[0]["project"] == "airon-doctor"
    assert f"{server.url}/runs/{state.run_id}" in out
    assert "protocol version: 1" in out and "test-key" not in out


def test_doctor_without_any_configuration(capsys):
    assert run_cli("doctor") == 2
    out = capsys.readouterr().out
    assert "server address: not set" in out and "AIRON_URL" in out and "airon login" in out
    assert "API key: not set" in out and "AIRON_API_KEY" in out


def test_doctor_with_an_invalid_address(monkeypatch, capsys):
    monkeypatch.setenv("AIRON_URL", "ftp://x")
    monkeypatch.setenv("AIRON_API_KEY", "test-key")
    assert run_cli("doctor") == 2
    assert "not a valid http(s) address" in capsys.readouterr().out


def test_doctor_with_a_wrong_key_401(server, monkeypatch, capsys):
    configure(monkeypatch, server, key="wrong-wrong-wrong")
    assert run_cli("doctor") == 1
    out = capsys.readouterr().out
    assert "HTTP 401" in out and "does not accept this key" in out and "airon login" in out
    assert "wrong-wrong-wrong" not in out


def test_doctor_with_a_read_only_key_403(server, monkeypatch, capsys):
    configure(monkeypatch, server, key="ro-key")
    assert run_cli("doctor") == 1
    out = capsys.readouterr().out
    assert "HTTP 403" in out and "may not log runs" in out


def test_doctor_with_a_name_that_does_not_resolve(monkeypatch, capsys):
    def no_such_host(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_host)
    monkeypatch.setenv("AIRON_URL", "https://airon.example.com")
    monkeypatch.setenv("AIRON_API_KEY", "test-key")
    assert run_cli("doctor") == 1
    out = capsys.readouterr().out
    assert "cannot be resolved" in out and "AIRON_URL" in out


def test_doctor_when_nothing_listens(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.stop()
    assert run_cli("doctor") == 1
    assert "Nothing accepts connections" in capsys.readouterr().out


def test_doctor_with_https_against_a_plain_http_port(server, monkeypatch, capsys):
    monkeypatch.setenv("AIRON_URL", server.url.replace("http://", "https://"))
    monkeypatch.setenv("AIRON_API_KEY", "test-key")
    assert run_cli("doctor") == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and ("TLS" in out or "closed unexpectedly" in out)


def test_doctor_mentions_a_configured_proxy(monkeypatch, capsys):
    monkeypatch.setenv("AIRON_URL", "https://airon.example.com")
    monkeypatch.setenv("AIRON_API_KEY", "test-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert run_cli("doctor") == 1
    assert "proxy is configured through HTTPS_PROXY" in capsys.readouterr().out


def test_doctor_with_a_wrong_path(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.inject("GET /v1/ping", 404)
    assert run_cli("doctor") == 1
    assert "not an Airon server" in capsys.readouterr().out


def test_doctor_with_a_redirect(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.inject("GET /v1/ping", 301)
    assert run_cli("doctor") == 1
    assert "redirects" in capsys.readouterr().out


def test_doctor_with_a_server_that_is_not_json(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.garbage = True
    assert run_cli("doctor") == 1
    assert "not with JSON" in capsys.readouterr().out


def test_doctor_with_a_server_error(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.inject("GET /v1/ping", 502)
    assert run_cli("doctor") == 1
    assert "internal error" in capsys.readouterr().out


def test_doctor_with_a_protocol_mismatch(server, monkeypatch, capsys):
    configure(monkeypatch, server)
    server.protocol = 2
    assert run_cli("doctor") == 1
    out = capsys.readouterr().out
    assert "server reports 2" in out and "upgrade" in out


def test_doctor_reports_waiting_spooled_runs(server, start, monkeypatch, capsys):
    run = start(mode="offline")
    run.finish()
    configure(monkeypatch, server)
    assert run_cli("doctor") == 0
    assert "1 spooled run(s) are waiting" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (Result("retry", kind="timeout"), "did not answer in time"),
        (Result("retry", kind="tls", message="certificate verify failed"), "SSL_CERT_FILE"),
        (Result("retry", kind="proxy", message="407"), "HTTPS_PROXY"),
        (Result("retry", kind="reset"), "closed unexpectedly"),
        (Result("fatal", 426), "protocol version 1"),
        (Result("drop", 418), "Unexpected answer"),
    ],
)
def test_explanations_for_each_failure_kind(result, expected):
    assert expected in cli.explain(result, "https://airon.example.com")
