"""`airon init`: write a starting airon.yaml that explains itself to whoever fills it in.

The comments are the documentation an agent reads before running anything: what each stage is for,
what the dry run must prove, how checkpoints and resuming work with the SDK, and how to launch,
follow, retry and stop a job. The guesses (setup command, training script) come from the files in
the repository and are marked for checking.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE = """\
# airon.yaml: how Airon runs this repository's training on a GPU machine.
#
# HOW IT WORKS
#   You describe the work here; Airon rents the machine, the machine clones the pushed commit and
#   runs the stages below in a fixed order, reporting each one. It is switched off when the job is
#   over. You never start scripts by hand over SSH.
#
#     airon check                    validate this file (free, nothing runs)
#     git push                       the machine clones the commit, so it must be on the remote
#     airon launch                   rent the machine and start (billable from here)
#     airon status DY-40 --watch     one line per change; exits 0 done, 1 failed, 130 stopped
#     airon retry DY-40              after a failure: fix, push, retry (same machine if still kept)
#     airon stop DY-40               stop the job; its machine is switched off
#   Agents use the MCP tools of the same names: launch, wait_job, job_status, retry_job, stop_job.
#
# STAGES (fixed order; the first failure stops the job)
#   clone   Airon: the repository at the pushed commit, into ~/work/<job> on the machine.
#   setup   Install exactly what the code needs, from a lock file, so every run gets the same.
#   data    Fetch the datasets, at the revision recorded in Airon's catalogue (list `datasets:`).
#   check   Airon: the right GPUs are there, there is disk, the datasets are in the catalogue.
#   test    Fast tests (a few minutes at most). Leave it out if the repository has none.
#   dry     The train command with FEWER STEPS, NOTHING ELSE CHANGED: same batch size, same
#           sequence length, same model. It must log steps with the SDK, keep the loss finite and
#           save at least one checkpoint. Airon then runs it a second time with AIRON_RESUME set to
#           that checkpoint, and it must continue past it. This is where an out-of-memory error, a
#           NaN or a broken resume show up in two minutes instead of after an hour.
#   train   The full run. It must continue from a checkpoint when it is restarted.
#   eval    Final evaluation. Metrics logged as eval/... are copied into the experiment.
#   finish  Airon: the machine is free and is switched off.
#
# THE TRAINING SCRIPT, with the SDK (`pip install airon`; add it to the lock file):
#     import airon
#     path = airon.resume()             # a checkpoint to continue from, when Airon asks; else None
#     start = load(path) if path else 0
#     run = airon.init()                # inside a job: no arguments, it joins the job's run
#     for step in range(start, total):
#         run.log({{"train/loss": loss}}, step=step)
#         if step % 500 == 0:
#             run.checkpoint(save(step), step=step)   # a file or folder: copied, then uploaded to
#                                                     # Airon's store in the background
#
# SEEING WHAT EVERY STAGE DOES (data, test, a run of audio through a model, anything)
#   A tqdm bar needs nothing: Airon reads it from the log and shows how many are done, the speed,
#   the time left, and that a bar has stopped moving. Without tqdm:
#       for path in airon.track(files, "transcribe", unit="files"): ...
#       airon.metric(wer=0.142)          # any number in any stage; shown as <stage>/wer
#   or from bash: echo "::airon progress name=resample done=400 total=3000 unit=files"
#   `airon status DY-40 --watch` prints a line at every tenth of the way.
#
# SPLITTING THE WORK OVER SEVERAL MACHINES (resources.shards: 8)
#   Shard 0 starts alone; once its dry run has passed, shards 1-7 start and skip test and dry.
#   Each machine gets AIRON_SHARD (0-7) and AIRON_SHARDS (8); the script takes its part with
#       files = airon.shard(sorted(all_files))          # every 8th file, a different start each
#       airon.output("results/part.jsonl")              # kept in Airon's store, never deleted
#   A `merge` stage then runs once on one more machine: airon.outputs() lists every shard's
#   results and airon.fetch(uri) downloads one. A failed shard is retried alone
#   (airon retry DY-40/4.3) while the others go on; `airon status DY-40/4` shows every shard.
#
# WHEN SOMETHING FAILS
#   The cause is recorded (the exception, the exit code; 137 means killed for lack of memory, the
#   last error line). The machine is kept for on_fail.hold so you can look (`airon status` prints
#   its ssh command, for emergencies), and `airon retry` goes on there from the failed stage with
#   your new commit, without setting everything up again.
#   When AWS has no machine of the type, the job waits (queued) and starts by itself, for up to
#   resources.wait.
#
# Never put secrets here: the file is stored in Airon and shown to people. The machine gets
# GH_TOKEN, HF_TOKEN and the like from the pod secret.

experiment: {code}                 # a RUNNING experiment in Airon (create_experiment, then start it)

resources:
  gpu: g5.xlarge                   # instance type: g5.xlarge (A10G 24 GB), g7e.2xlarge (RTX PRO
                                   # 6000, 96 GB), p5.4xlarge (H100); `cloud_limits` shows what fits
  disk: 100                        # GB; the machine image alone takes 75
  ttl: 12h                         # Airon switches the machine off after this, whatever happens
  wait: 6h                         # how long to wait when no machine is free (0: fail at once)
  # shards: 8                      # split the work over 8 machines (see SPLITTING below)

stages:
  setup: {setup}{setup_note}
  # data: python -m data.fetch my-dataset          # uncomment if data must be downloaded
  # test: pytest -x -q tests/fast                   # uncomment if there are fast tests
  dry: {train} --max-steps 50{train_note}
  train: {train}
  # eval: python eval.py --ckpt best
  # merge: python merge.py                          # with shards: once, after every shard

datasets: []                       # names from Airon's dataset catalogue that this job reads

estimate: {{steps: 10000}}           # steps of the full train: the dry run's speed turns it into
                                   # time and cost, and Airon refuses a run that cannot fit in ttl

metrics: {{primary: eval/loss, goal: min}}   # REQUIRED: the eval/ metric this experiment is
                                   # judged by (eval/wer, eval/sisdr...) and whether lower (min)
                                   # or higher (max) is better. The dry run must log it, with
                                   # train/loss and train/lr: run the evaluation on a small slice

checkpoint: {{every: 30m}}           # save at least this often (run.checkpoint): each one goes to
                                   # Airon's store, where any machine can read it; a dead machine
                                   # loses no more than this. `from: DY-40` (or DY-40/2, or an
                                   # s3:// address) starts this job from another one's checkpoint

on_fail: {{hold: 30m}}               # keep the machine this long after a failure, for a retry
quiet_ok: 20m                      # no output and an idle GPU for this long counts as stuck
"""


RUN_TEMPLATE = """\
# airon.yaml: a job that does not train (type: run): download, process, transcribe, compare
# models. Airon rents the machine(s), clones the pushed commit, installs the dependencies from the
# lock file (uv.lock, requirements.txt or pyproject.toml), downloads the inputs, runs the command,
# uploads what it wrote to $AIRON_OUTPUT and switches the machine(s) off.
#
#     airon check                    validate this file (free, nothing runs)
#     git push                       the machine clones the commit, so it must be on the remote
#     airon launch                   rent and start (billable from here)
#     airon status DY-40 --watch     follow it; exits 0 done, 1 failed, 130 stopped
#     airon retry DY-40 / airon stop DY-40
#
# WHAT THE SCRIPT GETS (environment variables)
#   $AIRON_DATA      the dataset of `inputs.data`, already downloaded. With shards, only this
#                    machine's part of it: use every file there, do not split it again.
#   $AIRON_MODELS    the best checkpoint of each experiment in `inputs.models`, in a folder each
#                    ($AIRON_MODELS/DY-40/...)
#   $AIRON_OUTPUT    an empty folder: write the results there; every file is uploaded, with its
#                    path, when the command ends without an error
#   $AIRON_SHARD, $AIRON_SHARDS   this machine's number and how many there are
#
# PROGRESS: a tqdm bar is read by itself; otherwise
#   for path in airon.track(files, "transcribe", unit="files"): ...
# Never put secrets here; HF_TOKEN and GH_TOKEN come from the pod secret.

experiment: {code}                 # a RUNNING experiment in Airon
type: run

gpu: g5.xlarge                     # instance type; `cloud_limits` shows what fits
disk: 200                          # GB; the image takes 75, and the inputs are downloaded here
ttl: 12h                           # the machine is switched off after this, whatever happens
# shards: 20                       # split the work over 20 machines: shard 0 first (with dry),
                                   # the rest once it passed

inputs:
  data: CHANGE-ME                  # a dataset from Airon's catalogue (list_datasets)
  # models: [DY-40, DY-41]         # best checkpoints of these experiments

dry: {python} {script} --limit 50{note}
run: {python} {script}

result:
  dataset: CHANGE-ME               # the uploaded results become this dataset in the catalogue,
                                   # so a later job can take them with `inputs: {{data: ...}}`
"""


def _setup(root: Path) -> tuple[str, str]:
    if (root / "uv.lock").exists():
        return "pip install -q uv && uv sync --frozen", ""
    for name in ("requirements.lock", "requirements.txt"):
        if (root / name).exists():
            return f"python3 -m venv .venv && .venv/bin/pip install -q -r {name}", ""
    if (root / "pyproject.toml").exists():
        return (
            "python3 -m venv .venv && .venv/bin/pip install -q -e .",
            "   # CHECK: no lock file found; pin versions",
        )
    return (
        "python3 -m venv .venv && .venv/bin/pip install -q airon",
        "   # CHECK: no requirements found; install what the code needs",
    )


def _train(root: Path, setup: str) -> tuple[str, str]:
    python = "uv run python" if setup.startswith("pip install -q uv") else ".venv/bin/python"
    for candidate in ("train.py", "scripts/train.py", "src/train.py", "training/train.py"):
        if (root / candidate).exists():
            return f"{python} {candidate}", "   # CHECK: the same arguments as train, fewer steps"
    return f"{python} train.py", "   # CHECK: no train.py found; put the real command"


def render(root: Path, code: str | None, kind: str = "train") -> str:
    if kind == "run":
        python = "uv run python" if (root / "uv.lock").exists() else "python"
        script = next(
            (c for c in ("job.py", "run.py", "main.py", "scripts/run.py") if (root / c).exists()),
            None,
        )
        return RUN_TEMPLATE.format(
            code=code or "CHANGE-ME",
            python=python,
            script=script or "job.py",
            note="" if script else "   # CHECK: no job.py found; put the real command",
        )
    setup, setup_note = _setup(root)
    train, train_note = _train(root, setup)
    return TEMPLATE.format(
        code=code or "CHANGE-ME",
        setup=setup,
        setup_note=setup_note,
        train=train,
        train_note=train_note,
    )
