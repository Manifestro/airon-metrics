from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

from airon import _env, _system


@pytest.fixture
def bin_dir(tmp_path, monkeypatch):
    """A directory placed first on PATH, for fake executables."""
    path = tmp_path / "bin"
    path.mkdir()
    monkeypatch.setenv("PATH", f"{path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setitem(sys.modules, "pynvml", None)  # force the nvidia-smi route
    return path


def fake_smi(bin_dir, body):
    script = bin_dir / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)


@pytest.fixture
def no_gpu(tmp_path, monkeypatch):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setitem(sys.modules, "pynvml", None)


SMI_OUTPUT = """echo "0, NVIDIA H100 80GB HBM3, 96, 71234, 81559, 64, 612.40"
echo "1, NVIDIA H100 80GB HBM3, 12, 1024, 81559, [N/A], [Not Supported]"
"""


def test_nvidia_smi_output_becomes_the_protocol_names(bin_dir):
    fake_smi(bin_dir, SMI_OUTPUT)
    sampler = _system.SystemSampler(lambda values: None)
    values = sampler.sample()
    assert values["gpu0/util_pct"] == 96
    assert values["gpu0/mem_used_mb"] == 71234
    assert values["gpu0/mem_total_mb"] == 81559
    assert values["gpu0/temp_c"] == 64
    assert values["gpu0/power_w"] == 612.4
    assert values["gpu1/util_pct"] == 12
    assert "gpu1/temp_c" not in values and "gpu1/power_w" not in values  # [N/A] is skipped
    assert "disk/used_gb" in values and "disk/total_gb" in values


def test_nvidia_smi_query_is_the_documented_one(bin_dir, tmp_path):
    log = tmp_path / "args"
    fake_smi(bin_dir, f'echo "$@" > {log}\n{SMI_OUTPUT}')
    _system.GpuReader().read()
    args = log.read_text()
    assert "--query-gpu=" in args and "--format=csv,noheader,nounits" in args


def test_gpu_inventory_for_the_environment_record(bin_dir):
    fake_smi(bin_dir, SMI_OUTPUT)
    assert _system.gpu_inventory() == [
        {"index": 0, "name": "NVIDIA H100 80GB HBM3", "mem_total_mb": 81559},
        {"index": 1, "name": "NVIDIA H100 80GB HBM3", "mem_total_mb": 81559},
    ]


def test_no_gpu_is_not_an_error(no_gpu):
    reader = _system.GpuReader()
    assert reader.read() == [] and reader.read() == []
    assert _system.gpu_inventory() == []
    values = _system.SystemSampler(lambda values: None).sample()
    assert not [k for k in values if k.startswith("gpu")]
    assert "disk/total_gb" in values


def test_failing_nvidia_smi_is_given_up_on_after_a_few_tries(bin_dir, tmp_path):
    counter = tmp_path / "calls"
    fake_smi(bin_dir, f"echo x >> {counter}\nexit 9")
    reader = _system.GpuReader()
    for _ in range(12):
        assert reader.read() == []
    assert len(counter.read_text().split()) == 5  # then it stops spawning processes


def test_hanging_nvidia_smi_cannot_stall_the_sampler(bin_dir):
    fake_smi(bin_dir, "sleep 30")
    reader = _system.GpuReader(smi_timeout=0.3)
    started = time.time()
    assert reader.read() == []
    assert time.time() - started < 3


def test_garbage_from_nvidia_smi_is_ignored(bin_dir):
    fake_smi(bin_dir, 'echo "NVIDIA-SMI has failed because it couldn\'t communicate"')
    assert _system.GpuReader().read() == []


def test_pynvml_is_used_when_importable(monkeypatch):
    class Handle:
        pass

    nvml = types.ModuleType("pynvml")
    nvml.NVML_TEMPERATURE_GPU = 0
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetCount = lambda: 1
    nvml.nvmlDeviceGetHandleByIndex = lambda i: Handle()
    nvml.nvmlDeviceGetName = lambda h: b"Fake GPU"  # older versions return bytes
    nvml.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=55)
    nvml.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(
        used=2048 * 1024 * 1024, total=8192 * 1024 * 1024
    )
    nvml.nvmlDeviceGetTemperature = lambda h, kind: 70
    nvml.nvmlDeviceGetPowerUsage = lambda h: 250000
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    (gpu,) = _system.GpuReader().read()
    assert gpu == {
        "index": 0,
        "name": "Fake GPU",
        "util": 55,
        "temp": 70,
        "power": 250.0,
        "mem_used": 2048.0,
        "mem_total": 8192.0,
    }


def test_pynvml_partial_failures_only_lose_that_reading(monkeypatch):
    nvml = types.ModuleType("pynvml")
    nvml.NVML_TEMPERATURE_GPU = 0
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetCount = lambda: 1
    nvml.nvmlDeviceGetHandleByIndex = lambda i: object()

    def boom(*args):
        raise RuntimeError("not supported")

    nvml.nvmlDeviceGetName = boom
    nvml.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=10)
    nvml.nvmlDeviceGetMemoryInfo = boom
    nvml.nvmlDeviceGetTemperature = boom
    nvml.nvmlDeviceGetPowerUsage = boom
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    sampler = _system.SystemSampler(lambda values: None)
    values = sampler.sample()
    assert values["gpu0/util_pct"] == 10
    assert "gpu0/temp_c" not in values and "gpu0/mem_used_mb" not in values


def test_psutil_is_used_when_installed(monkeypatch, no_gpu):
    fake = types.ModuleType("psutil")
    fake.cpu_percent = lambda interval=None: 37.5
    fake.virtual_memory = lambda: types.SimpleNamespace(total=16 * 1024**3, available=10 * 1024**3)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(_system, "_cgroup_memory", lambda: None)
    values = _system.HostReader().read()
    assert values["cpu/util_pct"] == 37.5
    assert values["ram/total_mb"] == 16 * 1024 and values["ram/used_mb"] == 6 * 1024


def test_proc_files_are_used_without_psutil(monkeypatch, no_gpu):
    stats = iter(
        [
            "cpu  100 0 100 800 0 0 0 0 0 0\n",  # when the reader starts
            "cpu  200 0 200 1000 0 0 0 0 0 0\n",  # 200 busy of 400 since then
        ]
    )
    meminfo = "MemTotal: 16384000 kB\nMemFree: 1000 kB\nMemAvailable: 8192000 kB\n"
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/stat":
            return io.StringIO(next(stats))
        if path == "/proc/meminfo":
            return io.StringIO(meminfo)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(_system, "open", fake_open, raising=False)
    monkeypatch.setattr(_system, "_cgroup_memory", lambda: None)
    reader = _system.HostReader()
    values = reader.read()
    assert values["cpu/util_pct"] == 50.0
    assert values["ram/total_mb"] == 16000 and values["ram/used_mb"] == 8000


def test_container_memory_limit_beats_the_hosts_numbers(monkeypatch, no_gpu):
    fake = types.ModuleType("psutil")
    fake.cpu_percent = lambda interval=None: 1.0
    fake.virtual_memory = lambda: types.SimpleNamespace(
        total=512 * 1024**3, available=500 * 1024**3
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(_system, "_cgroup_memory", lambda: (64 * 1024**3, 8 * 1024**3))
    values = _system.HostReader().read()
    assert values["ram/total_mb"] == 64 * 1024 and values["ram/used_mb"] == 8 * 1024


def test_a_missing_proc_does_not_raise(monkeypatch, no_gpu):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(_system, "_proc_cpu_times", lambda: None)
    monkeypatch.setattr(_system, "_proc_meminfo", lambda: None)
    monkeypatch.setattr(_system, "_cgroup_memory", lambda: None)
    values = _system.HostReader().read()
    assert set(values) <= {"disk/used_gb", "disk/total_gb"}


def test_run_sends_system_events_and_gpu_inventory(server, start, bin_dir):
    fake_smi(bin_dir, SMI_OUTPUT)
    run = start(system_metrics=True, system_interval=0.1)
    assert server.wait_for(lambda: [e for e in server.event_list() if "system" in e], timeout=10)
    run.finish()
    system = next(e["system"] for e in server.event_list() if "system" in e)
    assert system["gpu0/util_pct"] == 96 and "disk/used_gb" in system
    gpus = server.only_run().creates[0]["env"]["gpus"]
    assert gpus[0] == {"index": 0, "name": "NVIDIA H100 80GB HBM3", "mem_total_mb": 81559}


def test_run_without_a_gpu_still_reports_the_host(server, start, no_gpu):
    run = start(system_metrics=True, system_interval=0.1)
    assert server.wait_for(lambda: [e for e in server.event_list() if "system" in e], timeout=10)
    run.finish()
    keys = {k for e in server.event_list() if "system" in e for k in e["system"]}
    assert "disk/total_gb" in keys and not [k for k in keys if k.startswith("gpu")]
    assert "gpus" not in server.only_run().creates[0]["env"]


def test_sampler_failures_are_swallowed_and_sampling_continues(monkeypatch):
    calls = []

    def flaky(self):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("driver crashed")
        return {"cpu/util_pct": 1.0}

    monkeypatch.setattr(_system.SystemSampler, "sample", flaky)
    got = []
    sampler = _system.SystemSampler(got.append, interval=0.05)
    sampler.start()
    deadline = time.time() + 5
    while not got and time.time() < deadline:
        time.sleep(0.02)
    sampler.stop()
    assert got and len(calls) >= 2


# ---- environment capture ---------------------------------------------------------------------


def git(path, *args):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    subprocess.run(["git", *args], cwd=path, env=env, check=True, capture_output=True)


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@needs_git
def test_git_state_is_captured_and_remote_credentials_are_stripped(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    (repo / "a.txt").write_text("1")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "remote", "add", "origin", "https://user:hunter2token@example.com/acme/model.git")
    monkeypatch.chdir(repo)

    info = _env.capture()["git"]
    assert len(info["commit"]) >= 7 and info["branch"] == "main" and info["dirty"] is False
    assert info["remote"] == "https://example.com/acme/model.git"
    assert "hunter2token" not in json.dumps(info)

    (repo / "a.txt").write_text("2")
    assert _env.capture()["git"]["dirty"] is True


def test_outside_a_repository_there_is_no_git_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert "git" not in _env.capture()


def test_missing_git_binary_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert "python" in _env.capture()


def test_python_platform_and_host_are_always_present():
    info = _env.capture()
    assert info["python"].count(".") == 2 and info["platform"] and info["host"]


def test_only_already_imported_libraries_are_listed_and_none_is_imported(monkeypatch):
    for name in ("torch", "numpy", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.5.0+cu121"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    libs = _env.capture()["libs"]
    assert libs["torch"] == "2.5.0+cu121"
    assert "transformers" not in libs
    assert "transformers" not in sys.modules  # looking did not import anything


def test_library_version_falls_back_to_package_metadata(monkeypatch):
    bare = types.ModuleType("pytest_like")
    monkeypatch.setitem(sys.modules, "numpy", bare)  # no __version__ attribute
    versions = _env._lib_versions(["numpy"])
    assert versions == {} or isinstance(versions["numpy"], str)


def test_cloud_from_environment(monkeypatch):
    assert "cloud" not in _env.capture()
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123xyz")
    info = _env.capture()
    assert info["cloud"] == "runpod" and info["cloud_instance"] == "abc123xyz"
    monkeypatch.setenv("AIRON_CLOUD", "my-cluster")
    assert _env.capture()["cloud"] == "my-cluster"


def test_environment_variables_are_never_sent_wholesale(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hush-hush-1234")
    monkeypatch.setenv("HF_TOKEN", "hf_hushhush5678")
    monkeypatch.setenv("AIRON_API_KEY", "airon-key-91011")
    sent = json.dumps(_env.capture())
    for secret in ("hush-hush-1234", "hf_hushhush5678", "airon-key-91011"):
        assert secret not in sent


def test_capture_respects_its_time_budget(tmp_path, monkeypatch):
    slow = tmp_path / "git"
    slow.write_text("#!/bin/sh\nsleep 20\n")
    slow.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    started = time.time()
    _env.capture(budget_seconds=0.5)
    assert time.time() - started < 4
