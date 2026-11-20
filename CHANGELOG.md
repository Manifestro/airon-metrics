# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - 2026-11-20

Jobs: describe the work in `airon.yaml` and let `airon run` execute it stage by stage, so a
failure anywhere is reported to Airon with its cause. Needs an Airon server with jobs (`/v1/jobs`).

### Added

- `airon check`: the server validates `airon.yaml` without running anything.
- `airon run [--from STAGE]`: runs the fixed stages (clone, setup, data, check, test, dry, train,
  eval, finish) with `bash -eo pipefail`, each in its own process group, stops at the first
  failure and reports every stage. Output reaches the terminal unchanged and is sent to the run.
  The machine reports itself every 30 seconds and obeys a stop from Airon.
- The dry run runs a second time from the checkpoint it saved and must continue past it.
- A failed stage's cause names the exception, the exit code (137 or SIGKILL: likely out of memory)
  and the last line written to stderr.
- `airon status DY-40[/2]` and `airon stop DY-40[/2]`.
- `airon.resume()`: the checkpoint to continue from when Airon asks for one.

### Changed

- Inside a job, `airon.init()` attaches to the job's run and continues its numbering, does not
  capture the console (the runner sends it) and prints no information lines, only warnings.
- `console` defaults to `None`: on, except inside a job.

## [0.1.0] - 2026-10-31

First version, implementing wire protocol 1.

### Added

- `airon.init`, `run.log`, `run.progress`, `run.config`, `run.summary`, `run.checkpoint`,
  `run.finish`, the `with` form and the module-level `airon.log` and `airon.finish`.
- Non-blocking logging through a bounded queue and a background sender with batching, gzip,
  sequence numbers, retries with backoff and `Retry-After`, and the protocol's drop rules.
- Offline spool, automatic redelivery from the running process, and `airon sync`.
- Failure capture (`failed` with traceback, `killed` on `SIGTERM`) that chains existing hooks.
- Console tail capture that keeps carriage-return progress bars intact.
- Hardware sampling (NVML or `nvidia-smi`, `/proc` or `psutil`) and environment capture.
- `airon login`, `logout`, `doctor`, `sync` and `version`.
