"""HTTP client for the Airon protocol, built on the standard library only.

Every call returns a ``Result`` and never raises: the caller decides what a failure means.
"""

from __future__ import annotations

import gzip
import http.client
import json
import platform
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any

from . import _log
from ._version import PROTOCOL, __version__

GZIP_THRESHOLD = 1024
MAX_BODY_BYTES = 1024 * 1024
MAX_EVENTS_PER_REQUEST = 500

# Outcomes, from the "Retries" rule of the protocol.
OK = "ok"
RETRY = "retry"  # network error, 408, 425, 429, 5xx
DROP = "drop"  # any other 4xx: never retried
AUTH = "auth"  # 401 or 403: stop sending, keep data
SPLIT = "split"  # 413: send fewer events per request
FATAL = "fatal"  # redirect or 426: this server cannot be used as configured

USER_AGENT = f"airon/{__version__} python/{platform.python_version()}"


class Result:
    __slots__ = ("outcome", "status", "data", "retry_after", "message", "kind")

    def __init__(
        self,
        outcome: str,
        status: int | None = None,
        data: dict[str, Any] | None = None,
        retry_after: float | None = None,
        message: str = "",
        kind: str = "",
    ):
        self.outcome = outcome
        self.status = status
        self.data: dict[str, Any] = data if data is not None else {}
        self.retry_after = retry_after
        self.message = message
        self.kind = kind

    @property
    def ok(self) -> bool:
        return self.outcome == OK


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: they would replay the key to another address as a GET."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Map a network exception to (kind, short description)."""
    reason: Any = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        return "dns", "the host name could not be resolved"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "tls", f"the TLS certificate could not be verified ({reason.reason or reason})"
    if isinstance(reason, ssl.SSLError):
        return "tls", f"TLS error ({reason.reason or reason})"
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "timeout", "the server did not answer in time"
    if isinstance(reason, ConnectionRefusedError):
        return "refused", "the connection was refused"
    if isinstance(reason, (ConnectionResetError, BrokenPipeError, http.client.RemoteDisconnected)):
        return "reset", "the connection was closed by the other side"
    if isinstance(reason, OSError) and "Tunnel connection failed" in str(reason):
        return "proxy", f"the proxy refused the connection ({reason})"
    if isinstance(reason, (http.client.InvalidURL, ValueError)):
        return "url", f"the address is not valid ({reason})"
    if isinstance(reason, OSError):
        return "network", f"network error ({reason})"
    return "other", f"{type(reason).__name__}: {reason}"


def _retry_after(headers: Any) -> float | None:
    raw: Any = headers.get("Retry-After") if headers is not None else None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout
        _log.add_secret(api_key)
        self._opener: urllib.request.OpenerDirector | None = None

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        timeout: float | None = None,
    ) -> Result:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "X-Airon-Protocol": str(PROTOCOL),
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        payload: bytes | None = None
        if body is not None:
            try:
                payload = json.dumps(
                    body, separators=(",", ":"), allow_nan=False, default=str
                ).encode()
            except (ValueError, TypeError) as exc:
                return Result(DROP, message=f"could not encode the request: {exc}", kind="encode")
            if len(payload) > MAX_BODY_BYTES:
                return Result(SPLIT, message="the request is larger than 1 MiB", kind="size")
            headers["Content-Type"] = "application/json"
            if len(payload) > GZIP_THRESHOLD:
                payload = gzip.compress(payload, compresslevel=5, mtime=0)
                headers["Content-Encoding"] = "gzip"
        request = urllib.request.Request(
            self.base_url + path, data=payload, headers=headers, method=method
        )
        try:
            if self._opener is None:
                # Built on first use: creating the TLS context costs tens of milliseconds.
                self._opener = urllib.request.build_opener(_NoRedirect())
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                return self._from_response(response.status, response.headers, response.read())
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:
                raw = b""
            finally:
                exc.close()
            return self._from_response(exc.code, exc.headers, raw)
        except Exception as exc:
            kind, text = classify_exception(exc)
            return Result(RETRY, message=_log.redact(text), kind=kind)

    def _from_response(self, status: int, headers: Any, raw: bytes) -> Result:
        data: dict[str, Any] | None = None
        if raw.strip():
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                data = parsed
        retry_after = _retry_after(headers)
        if 200 <= status < 300:
            if data is None and raw.strip():
                # Captive portals and misconfigured proxies answer 200 with HTML.
                return Result(
                    RETRY,
                    status,
                    message="the answer was not JSON, is this an Airon server?",
                    kind="garbage",
                )
            return Result(OK, status, data)
        message = _error_message(status, data, headers)
        if status in (401, 403):
            return Result(AUTH, status, data, message=message)
        if status == 413:
            return Result(SPLIT, status, data, message=message)
        if status in (408, 425, 429) or status >= 500:
            return Result(RETRY, status, data, retry_after, message, kind="http")
        if 300 <= status < 400 or status == 426:
            return Result(FATAL, status, data, message=message)
        return Result(DROP, status, data, message=message)

    def ping(self, timeout: float | None = None) -> Result:
        return self.request("GET", "/v1/ping", timeout=timeout)

    def create_run(self, payload: dict[str, Any]) -> Result:
        result = self.request("POST", "/v1/runs", payload)
        if result.outcome in (DROP, SPLIT):
            # 404, 409 and 422 on create mean this run cannot be logged to as requested.
            result.outcome = FATAL
        return result

    def send_events(self, run_id: str, events: list[dict[str, Any]]) -> Result:
        return self.request("POST", f"/v1/runs/{run_id}/events", {"events": events})

    def finish_run(self, run_id: str, payload: dict[str, Any]) -> Result:
        return self.request("POST", f"/v1/runs/{run_id}/finish", payload)


def _error_message(status: int, data: dict[str, Any] | None, headers: Any) -> str:
    detail = ""
    if data:
        for key in ("error", "message", "detail"):
            if isinstance(data.get(key), str):
                detail = data[key]
                break
    if 300 <= status < 400 and headers is not None and headers.get("Location"):
        detail = f"redirected to {headers.get('Location')}"
    text = f"HTTP {status}" + (f": {detail}" if detail else "")
    return _log.redact(text[:300])


def send_split(client: Client, run_id: str, events: list[dict[str, Any]]) -> Result:
    """Send events, halving the batch while the server (or the size limits) say it is too big.

    Retried requests are harmless because the server ignores sequence numbers it already has.
    """
    if len(events) > MAX_EVENTS_PER_REQUEST:
        return _send_halves(client, run_id, events)
    result = client.send_events(run_id, events)
    if result.outcome == SPLIT:
        if len(events) == 1:
            return Result(DROP, result.status, message="a single event is too large to send")
        return _send_halves(client, run_id, events)
    return result


def _send_halves(client: Client, run_id: str, events: list[dict[str, Any]]) -> Result:
    middle = len(events) // 2
    merged: dict[str, Any] = {"accepted": 0, "duplicates": 0, "rejected": []}
    for part in (events[:middle], events[middle:]):
        result = send_split(client, run_id, part)
        if not result.ok:
            return result
        merged["accepted"] += int(result.data.get("accepted") or 0)
        merged["duplicates"] += int(result.data.get("duplicates") or 0)
        merged["rejected"] += list(result.data.get("rejected") or [])
    return Result(OK, 200, merged)
