# Usage guide

## A custom training loop with evaluation

```python
import airon

run = airon.init(
    project="memory-layer",
    name="baseline-v2",  # optional; a friendly name such as swift-river-7 is generated
    config={"lr": 3e-5, "batch_size": 32},  # shown next to the run
    tags=["baseline"],
    total_steps=num_epochs * steps_per_epoch,
    total_epochs=num_epochs,
)

step = 0
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)
        run.log(
            {"train/loss": loss, "lr": lr_now()},
            step=step,
            epoch=step / steps_per_epoch,
            samples=len(batch),
            tokens=batch_tokens,
        )
        step += 1
        if step % eval_every == 0:
            run.log({"eval/loss": evaluate()}, step=step)

run.finish()
```

Conventions that the interface understands: `train/loss`, `eval/loss`, `train/*`, `eval/*`, `lr`,
`train/grad_norm`. Any other metric is shown too. Names may use `/` for groups, up to 200 characters.
A nested dict such as `{"train": {"loss": 0.4}}` is flattened to `train/loss`.

### Steps, epochs, samples and tokens

- If you leave out `step`, an internal counter is used: it starts at 0 and grows by one per `log`
  call. Passing a `step` moves the counter to `step + 1`, so mixing the two is safe. Pass `step`
  explicitly whenever you log less often than every step, and for evaluation, so that `train/loss`
  and `eval/loss` line up on the same axis.
- `epoch` may be fractional.
- `samples` and `tokens` are the amounts processed by **one training step**. With the measured steps
  per second the server shows samples and tokens per second. The rate comes from the step difference
  over the time difference between logs and is smoothed, so logging every 100 steps gives the same
  figure as logging every step.
- `run.progress(step=..., total_steps=..., epoch=..., total_epochs=...)` reports progress without
  metrics, for example when the totals are only known after the dataloader is built.

### Metric values

Python numbers, numpy and torch scalars, and one-element tensors (anything with a scalar `.item()`)
are accepted; airon never imports numpy or torch. `NaN` and infinity are sent as the strings
`"NaN"`, `"Infinity"` and `"-Infinity"`: the server records them as a signal and leaves them out of
charts. Anything else (strings, lists, multi-element tensors) is dropped with one warning.

## Using `with`

```python
with airon.init(project="p") as run:
    ...
```

A normal exit finishes the run. An exception marks it `failed` with the exception type, message and
traceback, then continues to propagate. `KeyboardInterrupt` marks it `killed`. `sys.exit(0)` is a
normal end, any other exit code is `failed`.

Without `with` the same happens for a script that dies from an unhandled exception (airon chains
`sys.excepthook`, calling yours first) and for a script that simply ends.

`SIGTERM`, which is what cloud providers send before taking a machine away, is reported as `killed`
after the program's own handler has run. Install your own `SIGTERM` handler **before** calling
`airon.init`, so airon can chain it. If you install one later, it replaces airon's. Handlers are
only installed from the main thread.

## Config and summary

```python
run.config["warmup_steps"] = 500  # sent immediately, merged into the run's config
run.config.update(seed=7)
run.summary["eval/loss.best"] = 0.431  # values you want to see on the run's page
```

Both behave like dicts. Keys that look like secrets are masked with `***`. The summary is also sent
when the run finishes.

## Checkpoints

```python
run.checkpoint("hf://acme/model@3f2a1c", step=12000, label="best", **{"eval/loss": 0.431})
run.checkpoint("s3://bucket/run-7/step-24000.pt")
```

A checkpoint is a pointer: airon records the URI (and optional metrics) and never uploads a file.
`step` defaults to the last step you logged.

## Resuming

```python
run = airon.init(project="memory-layer", run_id="baseline-v2-a", resume="allow")
```

`resume="never"` (default) fails if the run id is already finished, `"allow"` reopens a finished run
or creates a new one, `"must"` requires an existing run. The run id can also come from the
`AIRON_RUN_ID` variable, which is convenient for restarts on a new pod. When resuming, `init` waits
up to `init_timeout` seconds (default 5) for the server to answer, so that sequence numbers and the
implicit step counter continue after the server's last values. Pass `step` explicitly if you restart
from a checkpoint at a different step.

If the server cannot be reached at that moment, the run continues in offline mode. Make sure the
server is reachable when you resume; events numbered while it was not may be treated as duplicates.

## Offline and `airon sync`

Without a server address (or with `AIRON_MODE=offline`) a run is written to
`~/.cache/airon/runs/<run id>/`. Upload everything with `airon sync`; a run that looks like it is
still alive is skipped unless you pass `--force`. After a successful upload the directory is
removed. An online run that loses its connection uses the same directory and uploads by itself when
the connection returns.

## Disabling

- `AIRON_MODE=disabled`, or `airon.init(mode="disabled")`: every call is a cheap no-op, nothing is
  written or sent, no thread is started.
- In distributed training every rank except 0 gets the same kind of no-op run. Use
  `rank_zero_only=False` to change that.

## Advanced settings

Keyword-only arguments to `airon.init`, mostly useful in tests:

| Argument | Default | Meaning |
|---|---|---|
| `flush_interval` | 2.0 | Seconds between sends (a batch is also sent at `batch_size` events) |
| `batch_size` | 200 | Events per request (at most 500) |
| `heartbeat_interval` | 30.0 | Heartbeat when nothing else was sent |
| `system_interval` | 10.0 | Hardware sampling |
| `max_queue` | 100000 | In-memory events before the oldest are dropped |
| `request_timeout` | 10.0 | Seconds for one HTTP request |
| `retry_min_delay`, `retry_max_delay` | 1.0, 60.0 | Backoff bounds |
| `retry_attempts` | 4 | Tries per batch before it goes to disk |
| `shutdown_timeout` | 10.0 | Longest wait when finishing |
| `console_lines` | 200 | Console lines kept between sends |
| `init_timeout` | 5.0 | How long a resuming `init` waits for the server |

## Limits worth knowing

- Console capture sees what Python writes to `sys.stdout` and `sys.stderr` after `init`. Output from
  subprocesses and C extensions, and `logging` handlers created before `init`, are not captured.
- Hardware sampling needs `nvidia-smi` or `nvidia-ml-py` for GPUs. On macOS install `psutil` for
  CPU and RAM.
- `sys.exit(1)` outside a `with` block cannot be told apart from a normal end, because Python gives
  exit hooks no exit code. Use the `with` block or call `run.finish(status="failed")`.
