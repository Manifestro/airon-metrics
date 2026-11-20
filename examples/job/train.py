"""A toy training loop that behaves like a real one: it logs a loss, saves checkpoints and
continues from one when Airon asks (`airon.resume()`). Used by examples/job/airon.yaml."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import airon

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--fail-at", type=int, default=-1)
args = parser.parse_args()
total = args.max_steps or args.steps
folder = os.environ.get("CKPT_DIR", "/tmp/airon-example-ckpt")
os.makedirs(folder, exist_ok=True)

start = 0
resume = airon.resume()
if resume and os.path.exists(resume):
    with open(resume) as f:
        start = json.load(f)["step"]
    print(f"continuing from step {start}")

run = airon.init(total_steps=total)
for step in range(start + 1, start + total + 1):
    loss = 2.0 * math.exp(-step / 30) + 0.1
    run.log({"train/loss": loss}, step=step)
    if step == args.fail_at:
        raise RuntimeError("simulated crash in the training loop")
    if step % 10 == 0:
        path = f"{folder}/step-{step}.json"
        with open(path, "w") as f:
            json.dump({"step": step}, f)
        run.checkpoint(path, step=step)
        print(f"step {step} loss {loss:.3f} saved {path}", flush=True)
    time.sleep(0.05)
run.finish()
