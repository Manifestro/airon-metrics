# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0] - 2026-11-20

Jobs on machines Airon rents. Needs an Airon server that can launch jobs.

### Added

- `airon launch [file]`: sends `airon.yaml` with the pushed commit and the repository; Airon
  rents the machine, which runs the job and is switched off when it is over.
- `airon agent`: what such a machine runs. It fetches its job, clones the repository at the commit
  (a GitHub token from the pod secret, `GH_TOKEN`, reaches git through its environment only, and an
  SSH address becomes HTTPS), and runs the stages like `airon run`.
- `examples/job`: a job that exercises every stage.
- `airon status JOB --watch`: one line per change (a stage starting, passing or failing, a waiting
  job getting its machine) and an exit code when the job is over: 0 done, 1 failed, 130 stopped.
  Made to run in the background or under a monitor.
- `airon status` shows what a waiting job waits for, and the `ssh` command while its machine is up.
- `airon retry JOB [--from STAGE] [--commit C | --same-commit] [--file airon.yaml]`: run a failed
  or stopped job again from the failed stage, with the pushed commit of this checkout. While the
  failed job's machine is kept, the same machine goes on: its agent waits there after a failure,
  fetches the new commit into the same folder (the virtualenv and data stay) and continues.
- `airon init [--experiment DY-40]`: writes a starting `airon.yaml` whose comments explain the
  stages, the dry run, checkpoints and `airon.resume()`, and how to launch, follow, retry and stop;
  the setup command and the training script are guessed from the repository.

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
