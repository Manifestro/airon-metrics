"""Where settings come from: arguments, then environment variables, then the credentials file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODES = ("online", "offline", "disabled")
RESUME_MODES = ("never", "allow", "must")


def env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def env_flag(name: str) -> bool:
    return (env(name) or "").lower() in ("1", "true", "yes", "on")


def _home() -> Path:
    try:
        return Path.home()
    except Exception:
        return Path(tempfile.gettempdir())


def airon_dir() -> Path | None:
    value = env("AIRON_DIR")
    return Path(value).expanduser() if value else None


def spool_root() -> Path:
    """Directory holding one sub-directory per spooled run."""
    base = airon_dir() or (_home() / ".cache" / "airon")
    return base / "runs"


def credentials_path() -> Path:
    base = airon_dir() or (_home() / ".config" / "airon")
    return base / "credentials.json"


def load_credentials() -> dict[str, str]:
    try:
        data = json.loads(credentials_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def save_credentials(url: str, api_key: str) -> Path:
    """Write the credentials file readable by its owner only."""
    path = credentials_path()
    created = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created:
        os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"url": url, "api_key": api_key}, handle)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def delete_credentials() -> bool:
    try:
        credentials_path().unlink()
    except FileNotFoundError:
        return False
    return True


def normalize_url(raw: str | None) -> str | None:
    """Return ``scheme://host[:port][/prefix]`` without a trailing slash, or None if unusable."""
    if not raw:
        return None
    text = raw.strip()
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError:
        return None
    return f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")


class Connection:
    """The resolved server address and key, and where each came from."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        url_source: str,
        key_source: str,
        raw_url: str | None,
    ):
        self.url = url
        self.raw_url = raw_url
        self.api_key = api_key
        self.url_source = url_source
        self.key_source = key_source

    @property
    def complete(self) -> bool:
        return bool(self.url and self.api_key)


def resolve_connection(url: str | None = None, api_key: str | None = None) -> Connection:
    stored: dict[str, Any] | None = None

    def pick(argument: str | None, variable: str, field: str) -> tuple[str | None, str]:
        nonlocal stored
        if argument:
            return argument, "argument"
        if (value := env(variable)) is not None:
            return value, f"environment variable {variable}"
        if stored is None:
            stored = load_credentials()
        if stored.get(field):
            return stored[field], str(credentials_path())
        return None, "not set"

    raw_url, url_source = pick(url, "AIRON_URL", "url")
    key, key_source = pick(api_key, "AIRON_API_KEY", "api_key")
    return Connection(normalize_url(raw_url), key, url_source, key_source, raw_url)
