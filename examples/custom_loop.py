"""A synthetic training loop with periodic evaluation. No third-party imports.

Run it against a server:

    export AIRON_URL=https://airon.example.com
    export AIRON_API_KEY=...            # a key that may log runs
    python examples/custom_loop.py

or without any server (the run is saved on disk; upload it later with `airon sync`):

    AIRON_MODE=offline python examples/custom_loop.py
"""

from __future__ import annotations

import math
import random
import time

import airon

EPOCHS = 3
STEPS_PER_EPOCH = 200
BATCH_SIZE = 32
SEQUENCE_LENGTH = 128
EVAL_EVERY = 100
LEARNING_RATE = 3e-4


def train_step(step: int) -> float:
    """Pretend to train: a noisy, decaying loss."""
    time.sleep(0.01)
    return 2.0 * math.exp(-step / 300) + random.random() * 0.05


def evaluate(step: int) -> float:
    time.sleep(0.05)
    return 2.0 * math.exp(-step / 300) + 0.1


def main() -> None:
    total_steps = EPOCHS * STEPS_PER_EPOCH
    config = {"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "seq_len": SEQUENCE_LENGTH}

    # The block reports "failed" with the traceback if the loop raises.
    with airon.init(
        project="examples",
        name="synthetic-loop",
        config=config,
        total_steps=total_steps,
        total_epochs=EPOCHS,
        tags=["example"],
    ) as run:
        best = float("inf")
        step = 0
        for _epoch in range(EPOCHS):
            for _ in range(STEPS_PER_EPOCH):
                loss = train_step(step)
                run.log(
                    {"train/loss": loss, "lr": LEARNING_RATE},
                    step=step,
                    epoch=step / STEPS_PER_EPOCH,
                    samples=BATCH_SIZE,  # per step; gives samples_per_sec
                    tokens=BATCH_SIZE * SEQUENCE_LENGTH,
                )
                step += 1

                if step % EVAL_EVERY == 0:
                    eval_loss = evaluate(step)
                    run.log({"eval/loss": eval_loss}, step=step)
                    if eval_loss < best:
                        best = eval_loss
                        run.summary["eval/loss.best"] = best
                        # A pointer only: Airon never receives the file itself.
                        run.checkpoint(
                            f"file:///checkpoints/step-{step}.pt", step=step, label="best"
                        )

        run.summary["train/final_loss"] = loss


if __name__ == "__main__":
    main()
