"""Describe the machine and code a run started on. Best effort: nothing here may raise or stall."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any

from ._config import env
from ._system import gpu_inventory

# Imported module name -> distribution name, when they differ.
_KNOWN_LIBS = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "transformers": "transformers",
    "datasets": "datasets",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "deepspeed": "deepspeed",
    "lightning": "lightning",
    "pytorch_lightning": "pytorch-lightning",
    "jax": "jax",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "triton": "triton",
    "flash_attn": "flash-attn",
    "xformers": "xformers",
    "bitsandbytes": "bitsandbytes",
    "safetensors": "safetensors",
    "diffusers": "diffusers",
    "vllm": "vllm",
}

# Environment variable that proves a provider, and the name we report for it.
_CLOUDS = (
    ("RUNPOD_POD_ID", "runpod"),
    ("VAST_CONTAINERID", "vast"),
    ("MODAL_TASK_ID", "modal"),
    ("PAPERSPACE_FQDN", "paperspace"),
    ("KAGGLE_KERNEL_RUN_TYPE", "kaggle"),
    ("COLAB_GPU", "colab"),
    ("SM_TRAINING_ENV", "sagemaker"),
    ("KUBERNETES_SERVICE_HOST", "kubernetes"),
)
_URL_CREDENTIALS = re.compile(r"^([a-z][a-z0-9+.-]*://)[^/@]+@", re.IGNORECASE)


def imported_libs() -> list[str]:
    """Names of well-known libraries the program has already imported (never imports any)."""
    return [name for name in _KNOWN_LIBS if name in sys.modules]


def _lib_versions(names: list[str]) -> dict[str, str]:
    libs: dict[str, str] = {}
    for name in names:
        version = getattr(sys.modules.get(name), "__version__", None)
        if not isinstance(version, str):
            try:
                version = importlib.metadata.version(_KNOWN_LIBS[name])
            except Exception:
                continue
        libs[_KNOWN_LIBS[name]] = version
    return libs


def _cloud() -> dict[str, str]:
    override = env("AIRON_CLOUD")
    if override:
        return {"cloud": override[:64]}
    for variable, name in _CLOUDS:
        value = os.environ.get(variable)
        if value is not None:
            info = {"cloud": name}
            if name == "runpod":
                info["cloud_instance"] = value[:128]
            return info
    return {}


class _Budget:
    """A shared time limit so a slow git or nvidia-smi cannot stall the run's start."""

    def __init__(self, seconds: float):
        self._end = time.monotonic() + seconds

    def left(self) -> float:
        return max(0.0, self._end - time.monotonic())


def _git(args: list[str], budget: _Budget) -> str | None:
    if budget.left() <= 0:
        return None
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=min(2.0, budget.left()),
            cwd=os.getcwd(),
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except Exception:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _git_info(budget: _Budget) -> dict[str, Any] | None:
    commit = _git(["rev-parse", "--short=12", "HEAD"], budget)
    if not commit:
        return None
    info: dict[str, Any] = {"commit": commit}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], budget)
    if branch:
        info["branch"] = branch
    status = _git(["status", "--porcelain", "--untracked-files=no"], budget)
    if status is not None:
        info["dirty"] = bool(status)
    remote = _git(["config", "--get", "remote.origin.url"], budget)
    if remote:
        # Remotes sometimes embed a token: https://user:token@example.com/...
        info["remote"] = _URL_CREDENTIALS.sub(r"\1", remote)
    return info


def capture(libs: list[str] | None = None, budget_seconds: float = 3.0) -> dict[str, Any]:
    """The ``env`` object of the create request."""
    budget = _Budget(budget_seconds)
    info: dict[str, Any] = {}
    for key, read in (
        ("python", platform.python_version),
        ("platform", platform.platform),
        ("host", socket.gethostname),
    ):
        try:
            info[key] = read()
        except Exception:
            continue
    info.update(_cloud())
    try:
        gpus = gpu_inventory()
        if gpus:
            info["gpus"] = gpus
    except Exception:
        pass
    git = _git_info(budget)
    if git:
        info["git"] = git
    versions = _lib_versions(imported_libs() if libs is None else libs)
    if versions:
        info["libs"] = versions
    return info
