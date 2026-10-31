# airon

Track machine learning runs from a plain Python training loop. `airon` sends metrics, progress and
hardware load to an [Airon](https://github.com/Manifestro/airon) server, where people watch loss,
ETA and GPU usage live.

- **One install, two variables, three lines.**
- **Never in the way of training.** Logging is a queue append; the network, the server and the disk
  cannot slow down or break your loop (see [the guarantees](#it-will-never-break-your-training)).
- **Built for rented GPUs.** Flaky networks, spot preemption and pods you cannot SSH into are the
  normal case: data that cannot be sent is kept on disk and uploaded later.
- **No required dependencies.** Python 3.9 to 3.13, standard library only.

## Install

```bash
pip install airon
```

Optional extras give better hardware readings: `pip install "airon[gpu]"` (NVML through
`nvidia-ml-py`) and `pip install "airon[system]"` (`psutil`, which also enables CPU and RAM on
macOS). Without them the SDK falls back to `nvidia-smi` and `/proc`.

## Set up

```bash
export AIRON_URL=https://airon.example.com
export AIRON_API_KEY=...        # a key that may log runs
```

Or run `airon login` once on a machine, which asks for both and saves them in
`~/.config/airon/credentials.json` (readable only by you). `airon doctor` checks the setup and tells
you what to fix. There is no default server: if nothing is configured, `airon.init` says so and
saves the run on disk instead of failing.

## Use

```python
import airon

with airon.init(
    project="memory-layer",
    name="baseline-v2",
    config={"lr": 3e-5, "batch_size": 32},
    total_steps=50_000,
    total_epochs=5,
) as run:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        run.log(
            {"train/loss": loss, "lr": scheduler.get_last_lr()[0]},
            step=step,
            epoch=step / steps_per_epoch,
            samples=32,
            tokens=32 * 512,
        )
        if step % 1000 == 0:
            run.log({"eval/loss": evaluate()}, step=step)
```

Leaving the `with` block normally marks the run `finished`; an exception marks it `failed` with the
traceback and is raised again. Without `with`, call `run.finish()` (or do nothing: a normal exit
finishes the run too). Metric values can be Python numbers, numpy or torch scalars and tensors with
one element; `NaN` and infinity are recorded, not dropped.

More patterns, such as eval loops, resuming, checkpoint pointers and disabling, are in
[docs/usage.md](docs/usage.md). Runnable examples: [`examples/custom_loop.py`](examples/custom_loop.py)
(no dependencies) and [`examples/pytorch_loop.py`](examples/pytorch_loop.py).

## What is collected automatically

| What | Details |
|---|---|
| Environment | Python and platform, host name, GPU names and memory, git commit, branch, dirty flag and remote (credentials in a remote URL are removed), versions of well-known libraries that your program has **already imported** (airon never imports torch itself), and the cloud when it can tell (`RUNPOD_POD_ID` gives `runpod`; override with `AIRON_CLOUD`) |
| Hardware, every 10 s | `gpu{i}/util_pct`, `mem_used_mb`, `mem_total_mb`, `temp_c`, `power_w`; `cpu/util_pct`; `ram/used_mb`, `ram/total_mb`; `disk/used_gb`, `disk/total_gb` for the working directory. Inside a container the RAM figures follow the container's memory limit |
| Progress | `steps_per_sec` (smoothed, from step deltas over time, so logging every 100 steps is fine), and `samples_per_sec` and `tokens_per_sec` when you pass `samples` and `tokens`. The server computes the ETA |
| Console | The last lines of stdout and stderr, with `\r` progress bars (tqdm) collapsed to their final state. Output is never swallowed. Turn off with `console=False` |
| Heartbeat | Every 30 s when nothing else was sent, so the server can tell a quiet run from a dead one |

Environment variables are never sent wholesale, and config values whose key looks like a secret
(`token`, `password`, `api_key`, ...) are replaced with `***`.

## Offline behaviour

Whatever cannot be delivered goes to JSON lines under `~/.cache/airon/runs/<run id>/`
(`$AIRON_DIR` changes the location). While the server is unreachable the running process keeps
trying and uploads the backlog, in order, as soon as it answers. After a crash or a long outage run
`airon sync` to upload what is left. `AIRON_MODE=offline` writes only to disk, for pods without
internet.

## Distributed training

With `torchrun` or similar only rank zero logs by default: processes with `RANK` or `LOCAL_RANK`
other than 0 receive a run object whose methods do nothing, so the same script works everywhere.
Pass `rank_zero_only=False` to log from every process.

## It will never break your training

The first requirement of this SDK, and each point has tests:

- `log()` is an append to a bounded in-memory queue: a few microseconds (median about 4 us on a
  laptop) under a lock that is held only for the append. No network or disk work happens in your
  thread; a background thread sends.
- Nothing raises into your code because of the network, the server or the disk: down, slow, wrong
  address, DNS failure, TLS errors, garbage replies, a read-only or full disk, and 401/403/409/5xx
  answers are all handled. Bad arguments to `init` or `log` produce at most one warning.
- At most one warning per kind of problem is printed, and the API key is removed from every message.
- Memory is bounded: at 100,000 queued events the oldest hardware samples go first, then the oldest
  metrics, with one warning.
- Retries use exponential backoff with jitter (1 s to 60 s) and honour `Retry-After`; retrying is
  safe because the server ignores sequence numbers it already has.
- At exit, flushing waits at most 10 seconds; whatever is left goes to disk.
- Your exception hook and `SIGTERM` handler are chained, never replaced: `failed` (with traceback) is
  reported after your hook has printed the error, and `killed` after your own handler has run.
- Threads are daemon threads; a forked child never reports for its parent.
- `AIRON_MODE=disabled` makes every call a no-op.

## Environment variables

| Variable | Meaning |
|---|---|
| `AIRON_URL` | Server address, for example `https://airon.example.com` |
| `AIRON_API_KEY` | The API key |
| `AIRON_PROJECT` | Default project (otherwise the name of the working directory) |
| `AIRON_RUN_ID` | Run to resume or attach to |
| `AIRON_MODE` | `online` (default), `offline` or `disabled` |
| `AIRON_DIR` | Where the spool and the credentials file live (default `~/.cache/airon` and `~/.config/airon`) |
| `AIRON_CLOUD` | Overrides the detected cloud name |
| `AIRON_QUIET` | `1` hides the start and finish lines (warnings still appear) |

Arguments to `airon.init` beat environment variables, which beat the credentials file.

## Command line

| Command | |
|---|---|
| `airon login` | Ask for the address and key, check them with the server, save them |
| `airon logout` | Delete the saved credentials |
| `airon doctor` | Check address, key, reachability and protocol version, write a tiny test run and print its link, with a fix for each failure |
| `airon sync` | Upload runs that were saved on disk |
| `airon version` | Print the version |

Exit codes: 0 success, 1 a check or an upload failed, 2 wrong usage or nothing configured.

## Links

- [Usage guide](docs/usage.md)
- [Wire protocol](docs/protocol.md), the contract with the server
- [Releasing](docs/releasing.md), [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Changelog](CHANGELOG.md)
- Server: [Manifestro/airon](https://github.com/Manifestro/airon)

## License

[MIT](LICENSE)
