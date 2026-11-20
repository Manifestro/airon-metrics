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

stages:
  setup: {setup}{setup_note}
  # data: python -m data.fetch my-dataset          # uncomment if data must be downloaded
  # test: pytest -x -q tests/fast                   # uncomment if there are fast tests
  dry: {train} --max-steps 50{train_note}
  train: {train}
  # eval: python eval.py --ckpt best

datasets: []                       # names from Airon's dataset catalogue that this job reads

estimate: {{steps: 10000}}           # steps of the full train: the dry run's speed turns it into
                                   # time and cost, and Airon refuses a run that cannot fit in ttl

checkpoint: {{every: 30m}}           # save at least this often (run.checkpoint): each one goes to
                                   # Airon's store, where any machine can read it; a dead machine
                                   # loses no more than this. `from: DY-40` (or DY-40/2, or an
                                   # s3:// address) starts this job from another one's checkpoint

on_fail: {{hold: 30m}}               # keep the machine this long after a failure, for a retry
quiet_ok: 20m                      # no output and an idle GPU for this long counts as stuck
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


def render(root: Path, code: str | None) -> str:
    setup, setup_note = _setup(root)
    train, train_note = _train(root, setup)
    return TEMPLATE.format(
        code=code or "CHANGE-ME",
        setup=setup,
        setup_note=setup_note,
        train=train,
        train_note=train_note,
    )
