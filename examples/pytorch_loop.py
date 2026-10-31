"""A small PyTorch model trained on random data, tracked with Airon.

    pip install airon torch
    python examples/pytorch_loop.py

Set AIRON_URL and AIRON_API_KEY to send the run to a server, or AIRON_MODE=offline to keep it
on disk. Torch is imported inside main() so a missing install gives a clear message.
"""

from __future__ import annotations

import sys

import airon

STEPS = 300
BATCH_SIZE = 64
FEATURES = 32
EVAL_EVERY = 100


def main() -> int:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("This example needs PyTorch: pip install torch", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    truth = torch.randn(FEATURES, 1, device=device)

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(BATCH_SIZE, FEATURES, device=device)
        return x, x @ truth

    with airon.init(
        project="examples",
        name="pytorch-regression",
        config={"lr": 1e-3, "batch_size": BATCH_SIZE, "device": device},
        total_steps=STEPS,
    ) as run:
        for step in range(1, STEPS + 1):
            x, y = batch()
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Tensors can be passed as they are: Airon calls .item() on them.
            run.log(
                {"train/loss": loss, "train/grad_norm": grad_norm},
                step=step,
                samples=BATCH_SIZE,
            )

            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    x, y = batch()
                    run.log({"eval/loss": loss_fn(model(x), y)}, step=step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
