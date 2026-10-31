"""Hardware sampling. Every reader degrades to "no data" instead of raising."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

MB = 1024 * 1024
GB = 1024 * 1024 * 1024
_SMI_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
_SMI_FAILURES_BEFORE_GIVING_UP = 5


def _number(text: Any) -> float | None:
    try:
        return float(str(text).strip())
    except ValueError:
        return None  # "[N/A]" and "[Not Supported]"


class GpuReader:
    """GPU readings through ``pynvml`` when importable, otherwise through ``nvidia-smi``."""

    def __init__(self, smi_timeout: float = 3.0):
        self._smi_timeout = smi_timeout
        self._nvml: Any = None
        self._handles: list[Any] = []
        self._smi_path: str | None = None
        self._backend = ""
        self._smi_failures = 0
        self._chosen = False

    def read(self) -> list[dict[str, Any]]:
        """One dict per GPU: index, name, util, mem_used, mem_total (MiB), temp, power (W)."""
        try:
            if not self._chosen:
                self._choose()
            if self._backend == "nvml":
                return self._read_nvml()
            if self._backend == "smi":
                return self._read_smi()
        except Exception:
            pass
        return []

    def _choose(self) -> None:
        self._chosen = True
        try:
            import pynvml  # type: ignore[import-not-found,import-untyped,unused-ignore]

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._nvml = pynvml
            self._backend = "nvml"
            return
        except Exception:
            self._nvml = None
        self._smi_path = shutil.which("nvidia-smi")
        self._backend = "smi" if self._smi_path else ""

    def _read_nvml(self) -> list[dict[str, Any]]:
        return [self._nvml_one(i, h) for i, h in enumerate(self._handles)]

    def _nvml_one(self, index: int, handle: Any) -> dict[str, Any]:
        nvml = self._nvml
        gpu: dict[str, Any] = {
            "index": index,
            "name": _try(lambda: _text(nvml.nvmlDeviceGetName(handle))),
            "util": _try(lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            "temp": _try(lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
            "power": _try(lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
        }
        memory = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
        if memory is not None:
            gpu["mem_used"] = memory.used / MB
            gpu["mem_total"] = memory.total / MB
        return gpu

    def _read_smi(self) -> list[dict[str, Any]]:
        try:
            done = subprocess.run(
                [
                    self._smi_path or "nvidia-smi",
                    f"--query-gpu={_SMI_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self._smi_timeout,
                check=True,
            )
        except Exception:
            self._smi_failures += 1
            if self._smi_failures >= _SMI_FAILURES_BEFORE_GIVING_UP:
                self._backend = ""
            return []
        self._smi_failures = 0
        gpus: list[dict[str, Any]] = []
        for row in csv.reader(done.stdout.splitlines(), skipinitialspace=True):
            if len(row) < 7:
                continue
            index = _number(row[0])
            gpus.append(
                {
                    "index": int(index) if index is not None else len(gpus),
                    "name": row[1].strip(),
                    "util": _number(row[2]),
                    "mem_used": _number(row[3]),
                    "mem_total": _number(row[4]),
                    "temp": _number(row[5]),
                    "power": _number(row[6]),
                }
            )
        return gpus


def _try(read: Callable[[], Any]) -> Any:
    try:
        return read()
    except Exception:
        return None


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


class HostReader:
    """CPU, RAM and disk. Uses psutil when installed, otherwise /proc on Linux."""

    def __init__(self, path: str | None = None):
        self._path = path
        self._psutil: Any = None
        try:
            import psutil  # type: ignore[import-untyped,unused-ignore]

            self._psutil = psutil
            psutil.cpu_percent(None)  # the first call only primes the counter
        except Exception:
            self._psutil = None
        self._last_cpu = _proc_cpu_times()

    def read(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in (self._cpu, self._ram, self._disk):
            try:
                out.update(part())
            except Exception:
                continue
        return out

    def _cpu(self) -> dict[str, float]:
        if self._psutil is not None:
            return {"cpu/util_pct": round(float(self._psutil.cpu_percent(None)), 1)}
        now = _proc_cpu_times()
        before, self._last_cpu = self._last_cpu, now
        if not now or not before:
            return {}
        busy, total = now[0] - before[0], now[1] - before[1]
        return {"cpu/util_pct": round(100.0 * busy / total, 1)} if total > 0 else {}

    def _ram(self) -> dict[str, float]:
        limit = _cgroup_memory()
        if self._psutil is not None:
            memory = self._psutil.virtual_memory()
            total, used = memory.total, memory.total - memory.available
        else:
            meminfo = _proc_meminfo()
            if not meminfo:
                return {}
            total, used = meminfo
        if limit is not None and limit[0] < total:
            total, used = limit  # inside a container the host's numbers would mislead
        return {"ram/used_mb": round(used / MB), "ram/total_mb": round(total / MB)}

    def _disk(self) -> dict[str, float]:
        usage = shutil.disk_usage(self._path or os.getcwd())
        return {
            "disk/used_gb": round(usage.used / GB, 2),
            "disk/total_gb": round(usage.total / GB, 2),
        }


def _proc_cpu_times() -> tuple[float, float] | None:
    """(busy, total) jiffies from /proc/stat."""
    try:
        with open("/proc/stat", encoding="ascii") as handle:
            fields = [float(x) for x in handle.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    if len(fields) < 5:
        return None
    total = sum(fields[:8])
    idle = fields[3] + fields[4]
    return total - idle, total


def _proc_meminfo() -> tuple[int, int] | None:
    """(total, used) bytes from /proc/meminfo."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                values[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    if "MemTotal" not in values:
        return None
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return values["MemTotal"], values["MemTotal"] - available


def _cgroup_memory() -> tuple[int, int] | None:
    """(limit, used) bytes of the container's memory cgroup, if it has a limit."""
    candidates = (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),  # cgroup v2
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    )
    for limit_file, used_file in candidates:
        try:
            limit = int(Path(limit_file).read_text().strip())
            used = int(Path(used_file).read_text().strip())
        except (OSError, ValueError):
            continue
        return limit, used
    return None


class SystemSampler:
    def __init__(self, emit: Callable[[dict[str, float]], None], interval: float = 10.0):
        self._emit = emit
        self._interval = max(0.05, interval)
        self._gpus = GpuReader()
        self._host = HostReader()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="airon-system", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def sample(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for gpu in self._gpus.read():
            prefix = f"gpu{gpu['index']}"
            for key, name in (
                ("util", "util_pct"),
                ("mem_used", "mem_used_mb"),
                ("mem_total", "mem_total_mb"),
                ("temp", "temp_c"),
                ("power", "power_w"),
            ):
                if gpu.get(key) is not None:
                    values[f"{prefix}/{name}"] = round(float(gpu[key]), 1)
        values.update(self._host.read())
        return values

    def _loop(self) -> None:
        # The first sample comes early so even short runs have a hardware point.
        wait = min(1.0, self._interval)
        while not self._stop.wait(wait):
            wait = self._interval
            try:
                values = self.sample()
                if values:
                    self._emit(values)
            except Exception:
                continue


def gpu_inventory() -> list[dict[str, Any]]:
    """GPU names and memory for the run's environment record."""
    return [
        {"index": g["index"], "name": g.get("name"), "mem_total_mb": round(g["mem_total"])}
        for g in GpuReader().read()
        if g.get("mem_total") is not None
    ]
