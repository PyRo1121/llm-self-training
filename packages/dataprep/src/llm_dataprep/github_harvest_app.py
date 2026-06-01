"""GitHub App installation tokens for dual-lane harvest (separate rate-limit pool).

Flow (https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app):
1. Sign JWT with app private key (``iss`` = Client ID, RS256, ≤10 min).
2. ``POST /app/installations/{installation_id}/access_tokens`` with ``Authorization: Bearer JWT``.
3. Use installation token for API calls (~1 hr lifetime); refresh before expiry.

Requires: ``pip install PyJWT cryptography`` (``uv sync --extra harvest``).
"""

from __future__ import annotations

import json
import os
import time
import warnings
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

API_ROOT = "https://api.github.com"
USER_AGENT = "llm-dataprep-github-harvest/0.1"
API_VERSION = "2026-03-10"


@dataclass(frozen=True)
class GitHubAppConfig:
    client_id: str
    installation_id: int
    private_key_pem: bytes
    download_only: bool = False


def app_config_from_env() -> GitHubAppConfig | None:
    """Return app config when Client ID, installation id, and private key are set."""
    client_id = os.environ.get("GITHUB_APP_CLIENT_ID", "").strip()
    if not client_id:
        return None
    inst_raw = os.environ.get("GITHUB_APP_INSTALLATION_ID", "").strip()
    if not inst_raw:
        return None
    try:
        installation_id = int(inst_raw)
    except ValueError:
        warnings.warn(
            f"GITHUB_APP_INSTALLATION_ID must be an integer, got: {inst_raw!r}; "
            "GitHub App lane disabled",
            stacklevel=2,
        )
        return None

    pem = _load_private_key_pem()
    if pem is None:
        return None

    download_only = os.environ.get("GITHUB_APP_DOWNLOAD_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return GitHubAppConfig(
        client_id=client_id,
        installation_id=installation_id,
        private_key_pem=pem,
        download_only=download_only,
    )


def _load_private_key_pem() -> bytes | None:
    inline = os.environ.get("GITHUB_APP_PRIVATE_KEY", "").strip()
    if inline:
        return inline.replace("\\n", "\n").encode("utf-8")
    path_raw = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "").strip()
    if not path_raw:
        return None
    path = Path(path_raw).expanduser()
    if not path.is_file():
        raise SystemExit(f"GITHUB_APP_PRIVATE_KEY_PATH not found: {path}")
    return path.read_bytes()


def make_app_jwt(*, client_id: str, private_key_pem: bytes, lifetime_s: int = 540) -> str:
    """Create a GitHub App JWT (max 10 minutes per docs)."""
    try:
        import jwt
    except ImportError as exc:
        raise SystemExit(
            "GitHub App auth requires PyJWT. Run: uv sync --package llm-dataprep --extra harvest"
        ) from exc

    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + min(600, max(60, lifetime_s)),
        "iss": client_id,
    }
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _app_api_request(
    method: str,
    url: str,
    *,
    jwt_token: str,
    body: dict | None = None,
) -> dict:
    data = None
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": API_VERSION,
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 404 and "/access_tokens" in url:
            raise RuntimeError(
                f"GitHub App API HTTP 404 for {url}: {detail}\n"
                "Hint: wrong GITHUB_APP_INSTALLATION_ID? "
                "Run: github-harvest --list-app-installations"
            ) from exc
        raise RuntimeError(f"GitHub App API HTTP {exc.code} for {url}: {detail}") from exc
    if not raw:
        return {}
    parsed = json.loads(raw.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def create_installation_access_token(
    *,
    client_id: str,
    private_key_pem: bytes,
    installation_id: int,
    permissions: dict[str, str] | None = None,
) -> tuple[str, float]:
    """Mint installation token; return (token, expires_at_epoch)."""
    jwt_token = make_app_jwt(client_id=client_id, private_key_pem=private_key_pem)
    url = f"{API_ROOT}/app/installations/{installation_id}/access_tokens"
    body: dict = {}
    if permissions:
        body["permissions"] = permissions
    doc = _app_api_request("POST", url, jwt_token=jwt_token, body=body or None)
    token = doc.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"Installation token response missing token: {doc!r}")
    expires_raw = doc.get("expires_at")
    expires_at = _parse_expires_at(expires_raw)
    return token, expires_at


def _parse_expires_at(raw: object) -> float:
    if isinstance(raw, str) and raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass
    return time.time() + 3600


def resolve_user_installation_id(
    *,
    client_id: str,
    private_key_pem: bytes,
    username: str,
) -> int:
    """``GET /users/{username}/installation`` using app JWT."""
    jwt_token = make_app_jwt(client_id=client_id, private_key_pem=private_key_pem)
    enc_user = urllib.parse.quote(username.strip(), safe="")
    url = f"{API_ROOT}/users/{enc_user}/installation"
    try:
        doc = _app_api_request("GET", url, jwt_token=jwt_token)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            raise RuntimeError(
                f"GitHub App is not installed on user {username!r}.\n"
                "Install it: GitHub → Settings → Applications → "
                "Installed GitHub Apps → Configure → find your app, OR open the app "
                "settings page → Install App → choose your account → Install.\n"
                "Then re-run: github-harvest --list-app-installations"
            ) from exc
        raise
    iid = doc.get("id")
    if not isinstance(iid, int):
        raise RuntimeError(
            f"No app installation found for user {username!r}. "
            "Install the app on your account first."
        )
    return iid


def list_installations(
    *,
    client_id: str,
    private_key_pem: bytes,
) -> list[dict]:
    jwt_token = make_app_jwt(client_id=client_id, private_key_pem=private_key_pem)
    url = f"{API_ROOT}/app/installations"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"GitHub App API HTTP {exc.code}: {detail}") from exc
    return data if isinstance(data, list) else []


class InstallationTokenProvider:
    """Callable token provider — refreshes installation token before expiry."""

    def __init__(self, cfg: GitHubAppConfig) -> None:
        self._cfg = cfg
        self._token: str | None = None
        self._expires_at: float = 0.0

    def __call__(self) -> str:
        if self._token and time.time() < self._expires_at - 120:
            return self._token
        token, expires_at = create_installation_access_token(
            client_id=self._cfg.client_id,
            private_key_pem=self._cfg.private_key_pem,
            installation_id=self._cfg.installation_id,
            permissions={"contents": "read", "metadata": "read"},
        )
        self._token = token
        self._expires_at = expires_at
        print(
            f"github-app: refreshed installation token (expires "
            f"{datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()})",
            flush=True,
        )
        return token


def token_provider_from_env() -> Callable[[], str] | None:
    cfg = app_config_from_env()
    if cfg is None:
        return None
    return InstallationTokenProvider(cfg)
