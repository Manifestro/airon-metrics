"""Checkpoints inside a job: copied at once, uploaded in the background through presigned URLs,
and downloaded again by `resume()` and `fetch()`. A fake Airon plays S3 too."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import airon


class Storage:
    """Airon's checkpoint endpoints, and the bucket behind its presigned URLs."""

    def __init__(self, part_size: int = 100 * 1024 * 1024) -> None:
        self.part_size = part_size
        self.objects: dict[str, bytes] = {}
        self.parts: dict[str, dict[int, bytes]] = {}
        self.recorded: list[dict[str, Any]] = []
        self.downloads = 0
        store = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, answer: dict[str, Any], status: int = 200) -> None:
                raw = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_PUT(self) -> None:  # noqa: N802  (a presigned upload URL)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                key, _, number = self.path.removeprefix("/put/").partition("?part=")
                store.parts.setdefault(key, {})[int(number)] = body
                self.send_response(200)
                self.send_header("ETag", f'"etag-{number}"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802  (a presigned download URL, or the outputs)
                if self.path == "/v1/job/outputs":
                    listed = [
                        {
                            "uri": f"s3://bucket/{r['key']}",
                            "name": r["key"].rsplit("/", 1)[-1],
                            "shard": 0,
                        }
                        for r in store.recorded
                        if r.get("label") == "output"
                    ]
                    raw = json.dumps({"outputs": listed}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                data = store.objects[self.path.removeprefix("/get/")]
                store.downloads += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                base = store.url
                if self.path == "/v1/job/checkpoint/start" and body.get("kind") == "output":
                    key = f"lab/XP-01/1/shard-0/outputs/{body['filename']}"
                    self._json(
                        {
                            "key": key,
                            "uri": f"s3://bucket/{key}",
                            "part_size": store.part_size,
                            "parts": [{"number": 1, "url": f"{base}/put/{key}?part=1"}],
                        }
                    )
                elif self.path == "/v1/job/checkpoint/start":
                    key = f"lab/XP-01/1/step-{body['step']:09d}/{body['filename']}"
                    count = max(1, -(-body["size"] // store.part_size))
                    self._json(
                        {
                            "key": key,
                            "uri": f"s3://bucket/{key}",
                            "upload_id": "u-1" if count > 1 else None,
                            "part_size": store.part_size,
                            "parts": [
                                {"number": n, "url": f"{base}/put/{key}?part={n}"}
                                for n in range(1, count + 1)
                            ],
                        }
                    )
                elif self.path == "/v1/job/checkpoint/finish":
                    key = body["key"]
                    parts = store.parts.pop(key)
                    assert [p["etag"] for p in body["parts"]] == [
                        f'"etag-{n}"' for n in sorted(parts)
                    ]
                    store.objects[key] = b"".join(parts[n] for n in sorted(parts))
                    store.recorded.append(body)
                    self._json({"uri": f"s3://bucket/{key}", "step": body["step"]})
                elif self.path in ("/v1/job/checkpoint/download", "/v1/checkpoints/download"):
                    key = body["uri"].removeprefix("s3://bucket/")
                    self._json({"uri": body["uri"], "url": f"{base}/get/{key}"})
                else:  # runs, events, finish: accepted and forgotten
                    self._json({})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Storage:
    made = Storage()
    monkeypatch.setenv("AIRON_URL", made.url)
    monkeypatch.setenv("AIRON_API_KEY", "job-key")
    monkeypatch.setenv("AIRON_JOB", "XP-01/1")
    monkeypatch.setenv("AIRON_STAGE", "train")
    monkeypatch.setenv("AIRON_PROJECT", "lab")
    yield made
    made.http.shutdown()


def test_a_checkpoint_is_copied_then_uploaded(storage: Storage, tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights at step 10")
    run = airon.init(system_metrics=False)
    run.checkpoint(str(weights), step=10)
    weights.write_bytes(b"the next save overwrote it at once")  # the copy is what goes up
    run.finish()  # waits for the upload
    key = "lab/XP-01/1/step-000000010/model.pt"
    assert storage.objects[key] == b"weights at step 10"
    assert storage.recorded[0]["step"] == 10 and storage.recorded[0]["label"] is None


def test_a_large_checkpoint_goes_up_in_parts(storage: Storage, tmp_path: Path) -> None:
    storage.part_size = 1024 * 1024
    big = tmp_path / "big.bin"
    data = bytes(range(256)) * (10 * 1024)  # 2.5 MiB: three parts
    big.write_bytes(data)
    run = airon.init(system_metrics=False)
    run.checkpoint(str(big), step=7)
    run.finish()
    assert storage.objects["lab/XP-01/1/step-000000007/big.bin"] == data
    assert storage.recorded[0]["upload_id"] == "u-1" and len(storage.recorded[0]["parts"]) == 3


def test_a_folder_checkpoint_comes_back_as_a_folder(
    storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "save_pretrained"
    (folder / "sub").mkdir(parents=True)
    (folder / "config.json").write_text('{"layers": 12}')
    (folder / "sub" / "weights.bin").write_bytes(b"w")
    run = airon.init(system_metrics=False)
    run.checkpoint(str(folder), step=3)
    run.finish()
    uri = "s3://bucket/lab/XP-01/1/step-000000003/save_pretrained.airon.tar"

    monkeypatch.setenv("AIRON_RESUME", uri)
    back = Path(str(airon.resume()))
    assert back.name == "save_pretrained" and back.is_dir()
    assert (back / "config.json").read_text() == '{"layers": 12}'
    assert (back / "sub" / "weights.bin").read_bytes() == b"w"
    assert airon.resume() == str(back) and storage.downloads == 1  # cached


def test_fetch_any_checkpoint_from_outside_a_job(
    storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.objects["other/AB-02/3/step-000000100/model.pt"] = b"someone else's weights"
    monkeypatch.delenv("AIRON_JOB")  # a laptop with `airon login`, not a job machine
    path = airon.fetch("s3://bucket/other/AB-02/3/step-000000100/model.pt")
    assert Path(path).read_bytes() == b"someone else's weights"
    assert airon.fetch("hf://org/model@rev") == "hf://org/model@rev"  # not Airon's: as it is


def test_dry_run_checkpoints_are_labelled(
    storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRON_STAGE", "dry")
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    run = airon.init(system_metrics=False)
    run.checkpoint(str(weights), step=20)
    run.finish()
    assert storage.recorded[0]["label"] == "dry"


def test_outside_a_job_a_checkpoint_is_only_a_pointer(tmp_path: Path) -> None:
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    run = airon.init(system_metrics=False, mode="offline")
    run.checkpoint(str(weights), step=1)
    run.finish()
    assert run._uploader is None


def test_resume_without_a_checkpoint_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIRON_RESUME", raising=False)
    assert airon.resume() is None
    monkeypatch.setenv("AIRON_RESUME", "/data/ckpt.pt")
    assert airon.resume() == "/data/ckpt.pt"


def test_without_storage_on_the_server_the_checkpoint_is_still_recorded(
    storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from airon import _checkpoints

    refused: list[str] = []
    original = _checkpoints.Uploader._upload

    def no_bucket(self: Any, item: dict[str, Any]) -> None:
        refused.append(item["source"])
        self.disabled = True
        raise RuntimeError("This Airon server keeps no checkpoints")

    monkeypatch.setattr(_checkpoints.Uploader, "_upload", no_bucket)
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    run = airon.init(system_metrics=False)
    pointers: list[dict[str, Any]] = []
    monkeypatch.setattr(run, "_event", lambda event: pointers.append(event))
    run.checkpoint(str(weights), step=5)
    run._uploader.wait(10)
    run.checkpoint(str(weights), step=6)  # not tried again: recorded at once
    assert refused == [str(weights)]
    assert [p["checkpoint"]["step"] for p in pointers] == [5, 6]
    assert pointers[0]["checkpoint"]["uri"] == str(weights)
    del original


def test_shard_takes_this_machines_part(monkeypatch: pytest.MonkeyPatch) -> None:
    files = [f"a{i:04d}.wav" for i in range(3000)]
    assert airon.shard_info() == (0, 1) and airon.shard(files) == files  # not split: everything
    monkeypatch.setenv("AIRON_SHARDS", "8")
    parts = []
    for i in range(8):
        monkeypatch.setenv("AIRON_SHARD", str(i))
        parts.append(airon.shard(files))
    assert sorted(f for part in parts for f in part) == files  # every file once
    assert {len(p) for p in parts} == {375}
    monkeypatch.setenv("AIRON_SHARD", "9")  # nonsense: treated as not split
    assert airon.shard_info() == (0, 1)


def test_outputs_are_stored_and_listed_for_the_merge(
    storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from airon import _checkpoints

    monkeypatch.setattr(_checkpoints, "_outputs", None)
    result = tmp_path / "part-0.jsonl"
    result.write_text('{"text": "hello"}\n')
    assert airon.output(str(result)) is True
    listed = airon.outputs()  # waits for the upload first
    assert [(o["name"], o["shard"]) for o in listed] == [("part-0.jsonl", 0)]
    assert storage.objects["lab/XP-01/1/shard-0/outputs/part-0.jsonl"] == b'{"text": "hello"}\n'
    assert Path(airon.fetch(listed[0]["uri"])).read_text() == '{"text": "hello"}\n'


def test_outputs_outside_a_job_do_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AIRON_JOB", raising=False)
    (tmp_path / "x").write_text("x")
    assert airon.output(str(tmp_path / "x")) is False
