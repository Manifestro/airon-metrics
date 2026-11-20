"""Progress for any stage: bars redrawn in place reach Airon while they run, and the helpers
print lines Airon reads."""

from __future__ import annotations

import io
from typing import Any

import pytest

import airon
from airon import _job


class Shipper:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, bool]] = []

    def add(self, stream: str, text: str, redraw: bool = False) -> None:
        self.lines.append((stream, text, redraw))

    def redraw(self, stream: str, text: str) -> None:
        self.add(stream, text, True)


def test_a_bar_redrawn_in_place_is_sent_before_its_line_ends() -> None:
    bar = b"\r 10%|#  | 1/10 [00:01<00:09, 1.0it/s]\r 20%|## | 2/10 [00:02<00:08, 1.0it/s]"
    ship = Shipper()

    class Source:  # what a pipe gives: the bar first, the end of the line later
        chunks = [bar, b"\ndone\n"]

        def read1(self, size: int) -> bytes:
            return self.chunks.pop(0) if self.chunks else b""

    _job._pump(Source(), io.BytesIO(), "stderr", ship)  # type: ignore[arg-type]
    assert ship.lines == [
        ("stderr", " 20%|## | 2/10 [00:02<00:08, 1.0it/s]", True),
        ("stderr", " 20%|## | 2/10 [00:02<00:08, 1.0it/s]", False),
        ("stderr", "done", False),
    ]


def test_redraws_are_thinned_and_tagged_with_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    shipper = _job.LogShipper(client=None)  # type: ignore[arg-type]
    shipper.stage = "data"
    now = [100.0]
    monkeypatch.setattr(_job.time, "time", lambda: now[0])
    for n, at in enumerate([100.0, 101.0, 106.0]):
        now[0] = at
        shipper.redraw("stderr", f"{n}/10")
    sent: list[dict[str, Any]] = shipper._lines
    assert [(line["text"], line["stage"], line["redraw"]) for line in sent] == [
        ("0/10", "data", True),
        ("2/10", "data", True),
    ]
    assert shipper.last_error == ""  # a bar is not why something failed


def test_track_and_metric_print_lines_inside_a_job(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AIRON_JOB", raising=False)
    assert list(airon.track([1, 2], "x")) == [1, 2]
    airon.metric(wer=0.1)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("AIRON_JOB", "XP-01/1")
    assert list(airon.track(["a", "b", "c"], "transcribe", unit="files", every=0)) == [
        "a",
        "b",
        "c",
    ]
    airon.metric(wer=0.142, rtf=float("nan"), note="x")
    airon.metric(step=3, loss=1.5)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "::airon progress name=transcribe done=0 unit=files total=3"
    assert out[-3] == "::airon progress name=transcribe done=3 unit=files total=3"
    assert out[-2] == "::airon metric wer=0.142"
    assert out[-1] == "::airon metric step=3 loss=1.5"
