from __future__ import annotations

import sys

import pytest

from airon._console import ConsoleCapture


def texts(capture):
    return [(line["stream"], line["text"]) for line in capture.drain()]


@pytest.fixture
def capture():
    capture = ConsoleCapture(max_lines=200)
    capture._active = True
    return capture


def test_complete_lines_are_captured_per_stream(capture):
    capture.feed("stdout", "hello\nworld\n")
    capture.feed("stderr", "oops\n")
    assert texts(capture) == [("stdout", "hello"), ("stdout", "world"), ("stderr", "oops")]


def test_progress_bars_collapse_to_their_final_state(capture):
    for percent in (0, 25, 50, 75, 100):
        capture.feed("stderr", f"\r{percent:3d}%|{'#' * (percent // 10):<10}| {percent}/100")
    capture.feed("stderr", "\n")
    assert texts(capture) == [("stderr", "100%|##########| 100/100")]


def test_a_bar_that_is_still_running_has_no_line_yet(capture):
    capture.feed("stderr", "\r 10%|#         |")
    capture.feed("stderr", "\r 20%|##        |")
    assert texts(capture) == []


def test_crlf_does_not_erase_the_line(capture):
    capture.feed("stdout", "windows style\r\nnext\r\n")
    assert texts(capture) == [("stdout", "windows style"), ("stdout", "next")]


def test_text_split_across_writes(capture):
    capture.feed("stdout", "par")
    capture.feed("stdout", "tial")
    capture.feed("stdout", " line\nsec")
    capture.feed("stdout", "ond\n")
    assert texts(capture) == [("stdout", "partial line"), ("stdout", "second")]


def test_cr_then_newline_in_separate_writes(capture):
    capture.feed("stdout", "done\r")
    capture.feed("stdout", "\n")
    assert texts(capture) == [("stdout", "done")]


def test_ansi_codes_and_blank_lines_are_removed(capture):
    capture.feed("stdout", "\x1b[32mgreen\x1b[0m\n\n   \nplain\n")
    assert texts(capture) == [("stdout", "green"), ("stdout", "plain")]


def test_only_the_tail_is_kept_and_the_gap_is_stated():
    capture = ConsoleCapture(max_lines=5)
    capture._active = True
    capture.feed("stdout", "".join(f"line {i}\n" for i in range(12)))
    lines = texts(capture)
    assert lines[0] == ("stderr", "[airon: 7 earlier lines omitted]")
    assert [t for _, t in lines[1:]] == [f"line {i}" for i in range(7, 12)]
    assert texts(capture) == []  # drained


def test_very_long_lines_are_truncated(capture):
    capture.feed("stdout", "x" * 100_000 + "\n")
    ((_, text),) = texts(capture)
    assert len(text) == 4000


def test_install_passes_everything_through_unchanged(capsys):
    capture = ConsoleCapture()
    capture.install()
    try:
        print("plain line")
        print("with\rcarriage return")
        sys.stderr.write("to stderr\n")
        sys.stdout.writelines(["a\n", "b\n"])
    finally:
        capture.uninstall()
    out = capsys.readouterr()
    assert out.out == "plain line\nwith\rcarriage return\na\nb\n"
    assert out.err == "to stderr\n"
    assert texts(capture) == [
        ("stdout", "plain line"),
        ("stdout", "carriage return"),
        ("stderr", "to stderr"),
        ("stdout", "a"),
        ("stdout", "b"),
    ]


def test_install_delegates_stream_attributes(capsys):
    capture = ConsoleCapture()
    capture.install()
    try:
        assert sys.stdout.isatty() is False or sys.stdout.isatty() is True
        assert hasattr(sys.stdout, "flush")
        sys.stdout.flush()
    finally:
        capture.uninstall()


def test_uninstall_restores_streams_and_flushes_a_partial_line(capsys):
    original = sys.stdout
    capture = ConsoleCapture()
    capture.install()
    assert sys.stdout is not original
    sys.stdout.write("no newline at the end")
    capture.uninstall()
    assert sys.stdout is original
    assert texts(capture) == [("stdout", "no newline at the end")]


def test_uninstall_does_not_clobber_a_stream_someone_else_wrapped(capsys):
    capture = ConsoleCapture()
    capture.install()
    mine = sys.stdout
    sys.stdout = other = type("Other", (), {"write": lambda self, s: len(s)})()
    try:
        capture.uninstall()
        assert sys.stdout is other
    finally:
        sys.stdout = mine
        capture.uninstall()


def test_capture_errors_never_reach_print(capsys, monkeypatch):
    capture = ConsoleCapture()
    capture.install()
    monkeypatch.setattr(capture, "feed", lambda *a: 1 / 0)
    try:
        print("still printed")
    finally:
        capture.uninstall()
    assert capsys.readouterr().out == "still printed\n"


def test_errors_from_the_real_stream_are_not_swallowed(capsys):
    class Broken:
        def write(self, text):
            raise BrokenPipeError("pipe closed")

    real = sys.stdout
    sys.stdout = Broken()
    capture = ConsoleCapture()
    capture.install()
    try:
        with pytest.raises(BrokenPipeError):
            print("x")
    finally:
        capture.uninstall()
        sys.stdout = real


def test_run_ships_console_lines_and_restores_streams(server, start, capsys):
    original = sys.stdout
    run = start(console=True, flush_interval=0.05)
    assert sys.stdout is not original
    print("epoch 1 done")
    for percent in (50, 100):
        sys.stderr.write(f"\r{percent}% |####|")
    sys.stderr.write("\n")
    print("last line without newline", end="")
    run.finish()
    assert sys.stdout is original
    lines = [line for e in server.event_list() if "log" in e for line in e["log"]]
    assert lines == [
        {"stream": "stdout", "text": "epoch 1 done"},
        {"stream": "stderr", "text": "100% |####|"},
        {"stream": "stdout", "text": "last line without newline"},
    ]
    assert "epoch 1 done" in capsys.readouterr().out  # and the user still saw it


def test_console_false_leaves_streams_alone(server, start):
    original = sys.stdout
    run = start(console=False)
    assert sys.stdout is original
    run.finish()
