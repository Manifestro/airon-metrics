from __future__ import annotations

import math
from pathlib import Path

from airon import _values


class FakeScalar:
    """Stands in for numpy and torch scalars: anything with a scalar ``.item()``."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeTensor:
    def item(self):
        raise ValueError("only one element tensors can be converted to Python scalars")


def collect(raw):
    warnings = []
    return _values.metrics(raw, lambda kind, text: warnings.append(kind)), warnings


def test_plain_numbers_pass_through():
    clean, warnings = collect({"a": 1, "b": 2.5, "c": True})
    assert clean == {"a": 1, "b": 2.5, "c": 1}
    assert warnings == []


def test_non_finite_values_become_the_protocol_strings():
    clean, _ = collect({"n": math.nan, "p": math.inf, "m": -math.inf})
    assert clean == {"n": "NaN", "p": "Infinity", "m": "-Infinity"}


def test_objects_with_item_are_unwrapped_without_importing_their_library():
    clean, warnings = collect(
        {"f": FakeScalar(0.5), "i": FakeScalar(7), "nan": FakeScalar(float("nan"))}
    )
    assert clean == {"f": 0.5, "i": 7, "nan": "NaN"}
    assert warnings == []


def test_item_that_fails_or_returns_junk_is_dropped_with_one_warning():
    clean, warnings = collect(
        {"tensor": FakeTensor(), "text": FakeScalar("x"), "s": "1.5", "none": None, "ok": 1}
    )
    assert clean == {"ok": 1}
    assert warnings == ["metric-value"] * 4  # the caller's warn_once prints it only once


def test_non_numeric_values_warn_once_through_the_real_logger(capsys):
    from airon import _log

    _log.reset()
    _values.metrics({"a": "x", "b": [1], "c": {}}, _log.warn_once)
    err = capsys.readouterr().err
    assert err.count("airon: warning") == 1


def test_nested_dicts_are_flattened_with_slashes():
    clean, _ = collect({"train": {"loss": 1.0, "acc": {"top1": 0.5}}})
    assert clean == {"train/loss": 1.0, "train/acc/top1": 0.5}


def test_bad_names_are_dropped():
    clean, warnings = collect({"": 1, "x" * 201: 1, "x" * 200: 2})
    assert list(clean.values()) == [2]
    assert warnings == ["metric-name", "metric-name"]


def test_metrics_that_is_not_a_dict_is_ignored():
    clean, warnings = collect([1, 2, 3])
    assert clean == {}
    assert warnings == ["metrics-type"]


def test_jsonable_handles_arbitrary_objects():
    class Odd:
        def __repr__(self):
            return "<odd>"

    value = {
        "path": Path("/tmp/x"),
        "tuple": (1, FakeScalar(2)),
        "odd": Odd(),
        "nan": float("nan"),
        1: "int key",
    }
    clean = _values.jsonable(value)
    assert clean == {
        "path": "/tmp/x",
        "tuple": [1, 2],
        "odd": "<odd>",
        "nan": "NaN",
        "1": "int key",
    }


def test_jsonable_survives_cycles():
    loop: dict = {}
    loop["me"] = loop
    assert isinstance(_values.jsonable(loop), dict)


def test_secrets_in_config_are_masked():
    clean = _values.scrub(
        {
            "lr": 1e-3,
            "hf_token": "abc",
            "nested": {"api_key": "k", "OPENAI_API_KEY": "k2"},
            "pw": "",
        }
    )
    assert clean["lr"] == 1e-3
    assert clean["hf_token"] == "***"
    assert clean["nested"] == {"api_key": "***", "OPENAI_API_KEY": "***"}
    assert clean["pw"] == ""
