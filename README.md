# airon

Track machine learning runs: send metrics from your training loop to an [Airon](https://github.com/Manifestro/airon) server and watch loss, progress, ETA and hardware load live.

> Status: early development. The wire protocol is specified in [docs/protocol.md](docs/protocol.md); the Python package is being written against it.

```python
import airon

with airon.init(project="my-project", name="baseline", total_steps=50_000, total_epochs=5) as run:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        run.log({"train/loss": loss}, step=step, epoch=epoch)
```

## Design goals

- **Never gets in the way of training.** Logging is non-blocking, network errors never reach your code, and metrics that cannot be sent are kept on disk and uploaded later.
- **Nothing to configure on a rented GPU.** Two environment variables (`AIRON_URL`, `AIRON_API_KEY`) and one `pip install`.
- **Zero required dependencies.** Hardware metrics use `nvidia-smi` and `/proc` when nothing better is installed.

## License

[MIT](LICENSE)
