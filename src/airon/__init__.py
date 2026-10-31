"""Airon: track machine learning runs without getting in the way of training.

import airon

run = airon.init(project="my-project", total_steps=50_000)
run.log({"train/loss": 0.4}, step=100)
run.finish()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _run
from ._run import NoopRun as NoopRun
from ._run import Run as Run
from ._run import finish as finish
from ._run import init as init
from ._run import log as log
from ._version import __version__ as __version__

if TYPE_CHECKING:
    run: Run | None


def __getattr__(name: str) -> Any:
    # ``airon.run`` is the current run, like ``wandb.run``; it changes on init and finish.
    if name == "run":
        return _run.current()
    raise AttributeError(f"module 'airon' has no attribute {name!r}")
