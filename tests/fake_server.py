"""A small Airon server for the tests, following docs/protocol.md.

It runs on a thread of the test process, keeps runs in memory, and can be told to misbehave:
inject status codes, add latency, answer with garbage, or be stopped and restarted.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

NON_FINITE = {"NaN", "Infinity", "-Infinity"}


@dataclass
class RunState:
    run_id: str
    status: str = "running"
    resumed: bool = False
    events: dict[int, dict[str, Any]] = field(default_factory=dict)
    creates: list[dict[str, Any]] = field(default_factory=list)
    finish: dict[str, Any] | None = None
    finish_count: int = 0
    last_step: int | None = None

    @property
    def last_seq(self) -> int:
        return max(self.events, default=0)


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    raw_size: int
    gzipped: bool
    body: Any
    at: float


class FakeServer:
    def __init__(
        self, keys: tuple[str, ...] = ("test-key",), read_only: tuple[str, ...] = ("ro-key",)
    ):
        self.keys = set(keys)
        self.read_only = set(read_only)
        self.runs: dict[str, RunState] = {}
        self.requests: list[Recorded] = []
        self.latency = 0.0
        self.garbage = False
        self.protocol = 1
        self.reject_names: set[str] = set()  # metric names the server refuses, with a reason
        self.lose_replies = 0  # process this many event batches, then hang up without answering
        self.max_body: int | None = None
        self.warnings: list[str] = []
        self._injected: list[tuple[str, int, float | None]] = []
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # ---- lifecycle ---------------------------------------------------------------------

    def start(self) -> FakeServer:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                server._handle(self, "GET")

            def do_POST(self) -> None:
                server._handle(self, "POST")

        ThreadingHTTPServer.allow_reuse_address = True
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def restart(self) -> None:
        self.stop()
        self.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ---- helpers for tests -------------------------------------------------------------

    def inject(
        self, match: str, status: int, times: int = 1, retry_after: float | None = None
    ) -> None:
        """Answer the next ``times`` requests whose "METHOD path" contains ``match`` with ``status``.

        A match starting with "=" must equal the whole "METHOD path" line.
        """
        with self._lock:
            self._injected.extend([(match, status, retry_after)] * times)

    def run(self, run_id: str) -> RunState:
        return self.runs[run_id]

    def only_run(self) -> RunState:
        assert len(self.runs) == 1, list(self.runs)
        return next(iter(self.runs.values()))

    def event_list(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None and not self.runs:
            return []
        state = self.run(run_id) if run_id else self.only_run()
        return [state.events[s] for s in sorted(state.events)]

    def calls(self, method: str, contains: str) -> list[Recorded]:
        return [r for r in self.requests if r.method == method and contains in r.path]

    def wait_for(self, predicate: Any, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.02)
        return bool(predicate())

    # ---- request handling ----------------------------------------------------------------

    def _handle(self, req: BaseHTTPRequestHandler, method: str) -> None:
        length = int(req.headers.get("Content-Length") or 0)
        raw = req.rfile.read(length) if length else b""
        gzipped = req.headers.get("Content-Encoding") == "gzip"
        body_bytes = gzip.decompress(raw) if gzipped and raw else raw
        try:
            body = json.loads(body_bytes) if body_bytes else None
        except ValueError:
            body = None
        with self._lock:
            self.requests.append(
                Recorded(
                    method,
                    req.path,
                    dict(req.headers.items()),
                    len(body_bytes),
                    gzipped,
                    body,
                    time.time(),
                )
            )
            line = f"{method} {req.path}"
            injected = next(
                (
                    i
                    for i in self._injected
                    if (i[0][1:] == line if i[0][0] == "=" else i[0] in line)
                ),
                None,
            )
            if injected:
                self._injected.remove(injected)
        if self.latency:
            time.sleep(self.latency)
        if injected:
            _, status, retry_after = injected
            headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
            self._reply(req, status, {"error": f"injected {status}"}, headers)
            return
        if self.garbage:
            self._reply(req, 200, raw_text="<html>captive portal</html>")
            return
        auth = req.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if key not in self.keys | self.read_only:
            self._reply(req, 401, {"error": "invalid key"})
            return
        path = req.path
        if method == "GET" and path == "/v1/ping":
            self._reply(
                req,
                200,
                {
                    "ok": True,
                    "protocol": self.protocol,
                    "key": {"name": "test", "scope": "tracking"},
                    "server_time": time.time(),
                },
            )
        elif method == "POST" and key in self.read_only:
            self._reply(req, 403, {"error": "read-only key"})
        elif method == "POST" and path == "/v1/runs":
            self._create(req, body)
        elif method == "POST" and path.startswith("/v1/runs/") and path.endswith("/events"):
            self._events(req, path.split("/")[3], body, len(body_bytes))
        elif method == "POST" and path.startswith("/v1/runs/") and path.endswith("/finish"):
            self._finish(req, path.split("/")[3], body)
        else:
            self._reply(req, 404, {"error": "not found"})

    def _create(self, req: BaseHTTPRequestHandler, body: Any) -> None:
        if not isinstance(body, dict) or not body.get("run_id") or not body.get("project"):
            self._reply(req, 422, {"error": "run_id and project are required"})
            return
        run_id, resume = body["run_id"], body.get("resume", "never")
        with self._lock:
            state = self.runs.get(run_id)
            if state is None:
                if resume == "must":
                    self._reply(req, 404, {"error": "no such run"})
                    return
                state = self.runs[run_id] = RunState(run_id)
                status = 201
            elif state.status == "running":
                status = 200
            elif resume in ("allow", "must"):
                state.status, state.resumed, status = "running", True, 200
            else:
                self._reply(req, 409, {"error": "run is over"})
                return
            state.creates.append(body)
            reply = {
                "run_id": run_id,
                "name": body.get("name") or "server-name",
                "url": f"{self.url}/runs/{run_id}",
                "status": "running",
                "resumed": state.resumed and status == 200,
                "last_step": state.last_step,
                "last_seq": state.last_seq,
                "warnings": list(self.warnings),
                "server_time": time.time(),
            }
        self._reply(req, status, reply)

    def _events(self, req: BaseHTTPRequestHandler, run_id: str, body: Any, size: int) -> None:
        with self._lock:
            state = self.runs.get(run_id)
        if state is None:
            self._reply(req, 404, {"error": "unknown run"})
            return
        events = body.get("events") if isinstance(body, dict) else None
        if not isinstance(events, list):
            self._reply(req, 422, {"error": "events must be a list"})
            return
        if len(events) > 500 or size > 1024 * 1024 or (self.max_body and size > self.max_body):
            self._reply(req, 413, {"error": "too large"})
            return
        accepted = duplicates = 0
        rejected: list[dict[str, Any]] = []
        with self._lock:
            for event in events:
                seq = event.get("seq") if isinstance(event, dict) else None
                if not isinstance(seq, int) or isinstance(seq, bool):
                    rejected.append({"seq": seq, "reason": "seq is not an integer"})
                    continue
                bad = next(
                    (
                        k
                        for k, v in (event.get("metrics") or {}).items()
                        if k in self.reject_names
                        or not (isinstance(v, (int, float)) or v in NON_FINITE)
                    ),
                    None,
                )
                if bad:
                    rejected.append({"seq": seq, "reason": f"metrics.{bad} is not a number"})
                    continue
                if seq in state.events:
                    duplicates += 1
                    continue
                state.events[seq] = event
                accepted += 1
                if event.get("step") is not None:
                    state.last_step = max(state.last_step or 0, event["step"])
            last_seq = state.last_seq
            lose = self.lose_replies > 0
            if lose:
                self.lose_replies -= 1
        if lose:
            req.close_connection = True
            req.connection.close()
            return
        self._reply(
            req,
            200,
            {
                "accepted": accepted,
                "duplicates": duplicates,
                "last_seq": last_seq,
                "rejected": rejected,
            },
        )

    def _finish(self, req: BaseHTTPRequestHandler, run_id: str, body: Any) -> None:
        with self._lock:
            state = self.runs.get(run_id)
            if state is None:
                self._reply(req, 404, {"error": "unknown run"})
                return
            state.finish_count += 1
            if state.finish is None or state.status == "running":
                state.finish = body
                state.status = (body or {}).get("status", "finished")
        self._reply(req, 200, {"ok": True})

    @staticmethod
    def _reply(
        req: BaseHTTPRequestHandler,
        status: int,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        raw_text: str | None = None,
    ) -> None:
        data = (raw_text if raw_text is not None else json.dumps(payload)).encode()
        try:
            req.send_response(status)
            req.send_header("Content-Type", "text/html" if raw_text else "application/json")
            req.send_header("Content-Length", str(len(data)))
            req.send_header("X-Airon-Protocol", "1")
            for key, value in (headers or {}).items():
                req.send_header(key, value)
            req.end_headers()
            req.wfile.write(data)
        except OSError:
            pass  # the client gave up (timeout); nothing to do
