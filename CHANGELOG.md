# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

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
