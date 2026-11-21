# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.6.1] - 2026-11-21

### Fixed

- An Airon address given with `/v1` at the end (`AIRON_URL=https://host/v1`) no longer becomes
  `/v1/v1/...`: the SDK drops it.
- The GitHub token is read from `GH_TOKEN` (or `GITHUB_TOKEN`); the old `GITHUB_API` name is gone.

## [0.6.0] - 2026-11-21

Jobs that do not train (`type: run`). Needs an Airon server that knows them; earlier servers
never send such a job, so this agent runs their jobs as before.

### Added

- A job of `type: run` (download, process, transcribe, compare models) runs its command in a new
  `run` stage instead of `train`: no checkpoint and no resume, and its `dry` is one short pass.
- Inputs: before the work, the agent downloads the job's `inputs` (a dataset from the catalogue,
  the best checkpoints of other experiments) from links Airon lists page by page, into
  `$AIRON_DATA` and `$AIRON_MODELS`. Files already on the disk are not downloaded again.
- Results: everything a `run` job writes to `$AIRON_OUTPUT` is uploaded when its command ends
  without an error, each file keeping its path, several at a time.
- Without a `setup` stage on a rented machine, the dependencies are installed from the lock file
  (uv.lock, requirements.txt or pyproject.toml) into the checkout's `.venv`, which every later
  stage uses.
- `airon init --type run` writes a starting file for such a job.

### Fixed

- A retry on a new machine prepared nothing when it started from a later stage: the code, its
  dependencies and the inputs are now prepared again; only the work before the start is skipped.

## [0.5.3] - 2026-11-21

### Added

- `run.log` warns once for every metric name that is not standard: `<section>/<name>` in lower
  case, sections train, eval, test, data, speed, sys (validation is eval/, not val/).
- `airon init` writes the required `metrics: {primary: eval/..., goal: min|max}` block.

## [0.5.2] - 2026-11-20

Progress of every stage, not only training. Needs an Airon server that reads it.

### Added

- `airon.track(items, name, unit=...)`, `airon.progress(name, done, total)` and
  `airon.metric(**values)`: progress and any numbers in any stage of a job (data preparation, a
  run through a model, tests), shown on the run as `<stage>/<name>`.
- `airon status` shows each stage's bars (done, speed, time left); `--watch` prints a line at
  every tenth of the way and when a bar stops moving.

### Changed

- `airon agent` sends a progress bar redrawn in place (tqdm) every 5 seconds while it runs, not
  only when its line ends, and tags every line with its stage. Python output is unbuffered, so
  prints arrive as they happen.

## [0.5.1] - 2026-11-20

### Fixed

- `airon agent` keeps asking for its job for up to 10 minutes after the machine boots, whatever
  the answer, instead of exiting on the first "unknown key". A machine that booted before Airon
  had recorded its job sat idle and paid for.

## [0.5.0] - 2026-11-20

Jobs split over several machines (`resources.shards`). Needs an Airon server with shards.

### Added

- `airon.shard(items)` and `airon.shard_info()`: this machine's part of the work in a split job.
- `airon.output(path)`: store a result of the job in Airon's store, kept for good; and
  `airon.outputs()`, every shard's results, for the `merge` stage.
- A `merge` stage, run once after every shard on its own machine.
- `airon status` lists every shard of a split job, and `--watch` reports each shard's changes;
  `airon retry DY-40/4` retries every failed shard, `airon retry DY-40/4.3` one.

### Changed

- A job's machine skips the stages Airon marks as done elsewhere: a later shard skips test and dry,
  which shard 0 proved.

## [0.4.1] - 2026-11-20

### Fixed

- A job's machine also finds its GitHub token under `GITHUB_API` (besides `GH_TOKEN` and
  `GITHUB_TOKEN`), the name pod secrets use, so private repositories clone.

## [0.4.0] - 2026-11-20

Checkpoints kept by Airon. Needs an Airon server with a checkpoint bucket; without one, checkpoints are recorded where they were saved, as before.

### Added

- Inside a job, `run.checkpoint(path)` copies the file or folder at once and uploads it in the background to Airon's checkpoint store through presigned URLs (in parts when large); the process waits for uploads before it ends (`AIRON_CHECKPOINT_WAIT`, default an hour). A failed upload, or a server without a store, still records the checkpoint where it was saved.
- `airon.resume()` downloads the checkpoint Airon asks the job to continue from and returns a local path (a folder comes back as a folder; cached). `airon.fetch(uri)` does the same for any stored checkpoint, also outside a job after `airon login`.
- A retried job, and a job with `checkpoint.from`, gets the checkpoint to continue from in `AIRON_RESUME` for its train stage and dry run.
- Dry-run checkpoints are labelled `dry`.

### Changed

- `airon init` writes `checkpoint: {every: 30m}`; `checkpoint.to` is no longer needed.

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
