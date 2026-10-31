# Contributing

Thanks for helping. The SDK has one rule above all others: **it must never break or slow down a
training run.** Changes that add work to `log()`, raise into user code, or block on the network are
not acceptable, whatever else they add.

## Setup

```bash
uv venv
uv pip install -e '.[dev]'
```

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

To try another Python: `uv run --python 3.9 --extra dev pytest`. The package supports 3.9 to 3.13, so
keep `from __future__ import annotations` at the top of modules and avoid syntax newer than 3.9 at
runtime.

## Tests

Tests start a fake Airon server (`tests/fake_server.py`, standard library only) that follows
[docs/protocol.md](docs/protocol.md) and can be told to misbehave. Nothing in the suite contacts a
real server. A change to behaviour needs a test that fails without it; a change to the protocol
needs a change to the document first.

## Style

- Standard library only at runtime. Optional packages (`nvidia-ml-py`, `psutil`) must be imported
  lazily and everything must work without them.
- Code that touches the network, the disk or the hardware catches its own failures.
- Comments explain why, not what.
- No real server addresses, keys or personal data anywhere, including tests and docs. Use
  `airon.example.com`.

## Pull requests

Describe the problem, show the test, and update `CHANGELOG.md` and the docs when behaviour changes.
