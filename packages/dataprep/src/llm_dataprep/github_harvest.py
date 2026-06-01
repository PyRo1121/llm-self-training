"""Harvest public agent session JSONL from GitHub code search (rate-limited).

Uses GITHUB_TOKEN from the environment (.env loaded by Makefile or shell).
Output: data/raw/public-github-sessions-YYYY-MM-DD.jsonl → scan-raw → curate.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from llm_core import data_dir, repo_root
from llm_core.paths import config_dir
from llm_dataprep.continue_sessions import _text_from_history_item
from llm_dataprep.openhands_events import _text_from_event
from llm_dataprep.cursor_transcripts import parse_line as parse_cursor_line
from llm_dataprep.cursor_transcripts import session_id_from_path
from llm_dataprep.gemini_cli import _messages_from_obj
from llm_dataprep.tokscale_cache import _role_text
from llm_dataprep.github_harvest_app import (
    app_config_from_env,
    list_installations,
    resolve_user_installation_id,
    token_provider_from_env,
)
from llm_dataprep.github_harvest_cache import HarvestCache
from llm_dataprep.github_harvest_graphql import BlobRequest, GraphQLBlobFetcher
from llm_dataprep.github_harvest_registry import (
    DEFAULT_EXCLUDE_PATH_REGEX,
    registry_queries,
)
from llm_dataprep.amp_threads import _text_from_message as _text_from_amp_message
from llm_dataprep.message_blocks import (
    USER_QUERY_RE,
    role_and_text_from_opencode,
    text_from_content_blocks,
)
from llm_dataprep.raw_io import append_records_buffered, dated_raw_path

API_ROOT = "https://api.github.com"
RAW_CDN_ROOT = "https://raw.githubusercontent.com"
USER_AGENT = "llm-dataprep-github-harvest/0.1"
GENERIC_ROLE_KEYS = ("role",)
GENERIC_TEXT_KEYS = ("text", "content", "message")
_COMMIT_REF_RE = re.compile(r"^https?://github\.com/[^/]+/[^/]+/blob/([^/]+)/")
_ROLLOUT_TRAILING_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.I,
)


class GitHubFetchError(RuntimeError):
    """Download failed; ``gone=True`` for 404 (file removed or stale index)."""

    def __init__(self, message: str, *, status: int | None = None, gone: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.gone = gone


def commit_ref_from_html_url(html_url: str) -> str:
    """Commit or branch ref from a code-search ``html_url`` (not the item blob sha)."""
    m = _COMMIT_REF_RE.match(html_url.strip())
    return m.group(1) if m else ""


@dataclass
class RateLimitSnapshot:
    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None
    resource: str | None = None


def _header(headers: dict[str, str], name: str) -> str | None:
    return headers.get(name.lower())


def _parse_rate_limit(headers: dict[str, str]) -> RateLimitSnapshot:
    rem = _header(headers, "X-RateLimit-Remaining")
    lim = _header(headers, "X-RateLimit-Limit")
    reset = _header(headers, "X-RateLimit-Reset")
    return RateLimitSnapshot(
        remaining=int(rem) if rem is not None else None,
        limit=int(lim) if lim is not None else None,
        reset_at=float(reset) if reset is not None else None,
        resource=_header(headers, "X-RateLimit-Resource"),
    )


def _backoff_seconds(
    attempt: int,
    *,
    retry_after: str | None,
    reset_at: float | None,
    base: float = 2.0,
    max_sleep: float = 300.0,
) -> float:
    """Exponential backoff with Retry-After / X-RateLimit-Reset preference."""
    if retry_after:
        try:
            return min(max_sleep, max(1.0, float(retry_after)))
        except ValueError:
            pass
    if reset_at is not None:
        return min(max_sleep, max(1.0, reset_at - time.time() + 1.0))
    return min(max_sleep, base * (2**attempt))


@dataclass
class HarvestConfig:
    max_file_bytes: int = 5_242_880
    max_files_per_run: int = 250
    max_lines_per_file: int = 50_000
    code_search_min_interval_s: float = 7.0
    download_min_interval_s: float = 0.05
    max_retries: int = 5
    low_remaining_threshold: int = 100
    code_search_low_remaining_threshold: int = 1
    use_raw_cdn: bool = True
    download_mode: str = "hybrid"  # hybrid | graphql | rest
    graphql_pending_flush: int = 16
    graphql_files_per_repo: int = 8
    graphql_max_repos: int = 3
    default_max_pages: int = 5
    max_search_requests_per_run: int | None = None
    redis_url: str | None = None
    state_path: Path = field(default_factory=lambda: data_dir() / "github_harvest/state.json")
    raw_prefix: str = "public-github-sessions"
    exclude_path_substrings: tuple[str, ...] = ()
    exclude_path_regex: tuple[str, ...] = ()
    exclude_repo_prefixes: tuple[str, ...] = ()
    queries: tuple[dict[str, Any], ...] = ()


@dataclass
class CodeHit:
    repo_full_name: str
    path: str
    sha: str  # Git blob OID from code search (content hash, not a commit ref)
    html_url: str
    query_id: str
    commit_ref: str = ""
    auth_lane: str = ""  # pat | app — credential that discovered this hit


def _bootstrap_env() -> None:
    """Load KEY=VALUE from repo .env / config/cloud.env if vars unset."""
    for rel in (".env", "config/cloud.env"):
        path = repo_root() / rel
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _optional_positive_int(raw: dict[str, Any], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    val = raw[key]
    if isinstance(val, bool):
        raise SystemExit(f"github-harvest config {key!r} must be an integer, got boolean")
    try:
        parsed = int(val)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"github-harvest config {key!r} must be an integer, got {val!r}"
        ) from exc
    return parsed


def load_harvest_config(path: Path | None = None) -> HarvestConfig:
    cfg_path = path or (config_dir() / "github-harvest.yaml")
    if not cfg_path.is_file():
        raise SystemExit(f"github-harvest config not found: {cfg_path}")
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = doc.get("github_harvest") or {}
    state = raw.get("state_path", "data/github_harvest/state.json")
    state_path = Path(state)
    if not state_path.is_absolute():
        state_path = repo_root() / state_path
    queries_raw = raw.get("queries")
    if queries_raw is None or queries_raw == "registry":
        enabled = raw.get("enabled_queries")
        disabled = raw.get("disabled_queries")
        queries = registry_queries(
            enabled=tuple(enabled) if enabled else None,
            disabled=tuple(disabled or ()),
        )
    else:
        queries = tuple(queries_raw or ())

    exclude_re = tuple(raw.get("exclude_path_regex") or ()) + DEFAULT_EXCLUDE_PATH_REGEX
    rate = raw.get("rate_limit") or {}

    return HarvestConfig(
        max_file_bytes=int(raw.get("max_file_bytes", 5_242_880)),
        max_files_per_run=int(raw.get("max_files_per_run", 250)),
        max_lines_per_file=int(raw.get("max_lines_per_file", 50_000)),
        code_search_min_interval_s=float(
            rate.get("search_min_interval_s", raw.get("code_search_min_interval_s", 7.0))
        ),
        download_min_interval_s=float(
            rate.get("download_min_interval_s", raw.get("download_min_interval_s", 0.05))
        ),
        max_retries=int(rate.get("max_retries", 5)),
        low_remaining_threshold=int(rate.get("low_remaining_threshold", 100)),
        code_search_low_remaining_threshold=int(
            rate.get("code_search_low_remaining_threshold", 1)
        ),
        use_raw_cdn=bool(rate.get("use_raw_cdn", True)),
        download_mode=str(rate.get("download_mode", raw.get("download_mode", "hybrid"))),
        graphql_pending_flush=int(rate.get("graphql_pending_flush", rate.get("graphql_batch_size", 16))),
        graphql_files_per_repo=int(rate.get("graphql_files_per_repo", 8)),
        graphql_max_repos=int(rate.get("graphql_max_repos", 3)),
        default_max_pages=int(raw.get("default_max_pages", 5)),
        max_search_requests_per_run=_optional_positive_int(
            raw, "max_search_requests_per_run"
        ),
        redis_url=(os.environ.get("REDIS_URL") or None),
        state_path=state_path,
        raw_prefix=str(raw.get("raw_prefix", "public-github-sessions")),
        exclude_path_substrings=tuple(raw.get("exclude_path_substrings") or ()),
        exclude_path_regex=exclude_re,
        exclude_repo_prefixes=tuple(raw.get("exclude_repo_prefixes") or ()),
        queries=queries,
    )


@dataclass
class HarvestLane:
    name: str
    client: GitHubClient
    gql: GraphQLBlobFetcher | None
    search: bool = True


def _make_client(
    token: str | Callable[[], str],
    cfg: HarvestConfig,
    *,
    lane: str,
) -> GitHubClient:
    client = GitHubClient(
        token,
        max_retries=cfg.max_retries,
        low_remaining_threshold=cfg.low_remaining_threshold,
        code_search_low_remaining_threshold=cfg.code_search_low_remaining_threshold,
        use_raw_cdn=cfg.use_raw_cdn,
        lane=lane,
    )
    return client


def _make_gql(
    token: str | Callable[[], str],
    cfg: HarvestConfig,
    client: GitHubClient,
) -> GraphQLBlobFetcher | None:
    if cfg.download_mode not in ("hybrid", "graphql"):
        return None
    return GraphQLBlobFetcher(
        token,
        max_retries=cfg.max_retries,
        on_rate_limit=client._apply_rate_limit_headers,
    )


def build_harvest_lanes(cfg: HarvestConfig, pat_token: str) -> list[HarvestLane]:
    """PAT lane + optional GitHub App installation lane (separate rate-limit pool)."""
    pat_client = _make_client(pat_token, cfg, lane="pat")
    lanes = [
        HarvestLane(
            name="pat",
            client=pat_client,
            gql=_make_gql(pat_token, cfg, pat_client),
            search=True,
        )
    ]
    app_cfg = app_config_from_env()
    if app_cfg is None:
        return lanes

    provider = token_provider_from_env()
    assert provider is not None
    app_client = _make_client(provider, cfg, lane="app")
    app_lane = HarvestLane(
        name="app",
        client=app_client,
        gql=_make_gql(provider, cfg, app_client),
        search=not app_cfg.download_only,
    )
    lanes.append(app_lane)
    if app_cfg.download_only:
        print("github-harvest: dual lane — PAT search + App downloads", flush=True)
    else:
        print("github-harvest: dual lane — PAT + App search (alternate each search page)", flush=True)
    return lanes


def pick_search_lane(lanes: list[HarvestLane], query_index: int) -> HarvestLane:
    search_lanes = [lane for lane in lanes if lane.search]
    if not search_lanes:
        raise RuntimeError("No search lane configured")
    return search_lanes[query_index % len(search_lanes)]


def lane_by_name(lanes: list[HarvestLane], name: str) -> HarvestLane:
    for lane in lanes:
        if lane.name == name:
            return lane
    return lanes[0]


def search_lanes(lanes: list[HarvestLane]) -> list[HarvestLane]:
    return [lane for lane in lanes if lane.search]


def pick_download_lane(lanes: list[HarvestLane], search_lane: HarvestLane) -> HarvestLane:
    for lane in lanes:
        if lane.name == "app" and not lane.search:
            return lane
    return search_lane


def _github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if not tok:
        raise SystemExit(
            "GITHUB_TOKEN is required. Add to .env or config/cloud.env "
            "(fine-grained PAT: public read / Contents read-only)."
        )
    return tok


class GitHubClient:
    def __init__(
        self,
        token: str | Callable[[], str],
        *,
        max_retries: int = 5,
        low_remaining_threshold: int = 100,
        code_search_low_remaining_threshold: int = 1,
        use_raw_cdn: bool = True,
        lane: str = "pat",
    ) -> None:
        self._token_provider: Callable[[], str] = (
            (lambda: token) if isinstance(token, str) else token
        )
        self._lane = lane
        self._max_retries = max(1, max_retries)
        self._low_remaining_threshold = low_remaining_threshold
        self._code_search_low_remaining_threshold = max(0, code_search_low_remaining_threshold)
        self._use_raw_cdn = use_raw_cdn
        self._last_code_search = 0.0
        self._last_download = 0.0
        self._rate_limits: dict[str, RateLimitSnapshot] = {}
        self._rate_limit = RateLimitSnapshot()

    def rate_limit_status(self) -> RateLimitSnapshot:
        return self._rate_limit

    def _bucket(self, *resources: str) -> RateLimitSnapshot:
        for res in resources:
            snap = self._rate_limits.get(res)
            if snap is not None and snap.remaining is not None:
                return snap
        return RateLimitSnapshot()

    def _auth_headers(self, *, accept: str | None = None, auth: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {"User-Agent": USER_AGENT}
        if auth:
            headers["Authorization"] = f"Bearer {self._token_provider()}"
        if accept:
            headers["Accept"] = accept
            headers["X-GitHub-Api-Version"] = "2026-03-10"
        return headers

    def _apply_rate_limit_headers(self, headers: dict[str, str]) -> None:
        snap = _parse_rate_limit(headers)
        if snap.remaining is not None:
            resource = snap.resource or "core"
            self._rate_limits[resource] = snap
            self._rate_limit = snap

    def _proactive_throttle(self, *, for_search: bool = False) -> None:
        if for_search:
            snap = self._bucket("code_search", "search")
            rem = snap.remaining
            reset_at = snap.reset_at
            if rem is None:
                return
            if rem <= self._code_search_low_remaining_threshold and reset_at is not None:
                sleep_s = max(1.0, reset_at - time.time() + 1.0)
                print(
                    f"github: code_search budget low ({rem} left) — sleeping {sleep_s:.0f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            return

        snap = self._bucket("core")
        rem = snap.remaining
        reset_at = snap.reset_at
        if rem is None:
            return
        if rem >= self._low_remaining_threshold:
            return
        if reset_at is None:
            time.sleep(2.0)
            return
        sleep_s = max(1.0, reset_at - time.time() + 1.0)
        print(
            f"github: core rate budget low ({rem}/{snap.limit} left) — sleeping {sleep_s:.0f}s",
            flush=True,
        )
        time.sleep(sleep_s)

    def _request(
        self,
        url: str,
        *,
        accept: str = "application/vnd.github+json",
        min_interval: float = 0.0,
        last_ts_attr: str | None = None,
        raw_bytes: bool = False,
    ) -> tuple[dict[str, Any] | list[Any] | bytes, dict[str, str]]:
        if min_interval > 0 and last_ts_attr:
            last = getattr(self, last_ts_attr, 0.0)
            wait = min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)

        for attempt in range(self._max_retries):
            is_search = "search/code" in url
            is_download = (
                url.startswith(RAW_CDN_ROOT)
                or "/git/blobs/" in url
                or "/contents/" in url
            )
            self._proactive_throttle(for_search=is_search)
            req = urllib.request.Request(
                url, headers=self._auth_headers(accept=accept, auth=not url.startswith(RAW_CDN_ROOT))
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    self._apply_rate_limit_headers(headers)
                    body = resp.read()
                    if last_ts_attr:
                        setattr(self, last_ts_attr, time.monotonic())
                    if raw_bytes or "application/vnd.github.raw" in accept:
                        return body, headers
                    if not body:
                        return {}, headers
                    parsed = json.loads(body.decode("utf-8"))
                    return parsed, headers
            except urllib.error.HTTPError as exc:
                err_headers = {k.lower(): v for k, v in exc.headers.items()}
                self._apply_rate_limit_headers(err_headers)
                if exc.code == 404 and is_download:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                    raise GitHubFetchError(
                        f"GitHub HTTP 404 for {url}: {detail}",
                        status=404,
                        gone=True,
                    ) from exc
                if exc.code in (403, 429, 502, 503) and attempt + 1 < self._max_retries:
                    exc.read()
                    sleep_s = _backoff_seconds(
                        attempt,
                        retry_after=err_headers.get("retry-after"),
                        reset_at=self._rate_limit.reset_at,
                    )
                    print(
                        f"github: HTTP {exc.code} — backoff {sleep_s:.0f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"GitHub HTTP {exc.code} for {url}: {detail}") from exc

        raise RuntimeError(f"GitHub request failed after {self._max_retries} retries: {url}")

    def code_search(self, query: str, *, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {"q": query, "page": page, "per_page": min(100, per_page)}
        )
        url = f"{API_ROOT}/search/code?{params}"
        data, hdr = self._request(
            url,
            accept="application/vnd.github.text-match+json",
            min_interval=0.0,
            last_ts_attr=None,
        )
        assert isinstance(data, dict)
        snap = self._bucket("code_search", "search")
        rem = snap.remaining
        if rem is not None and page == 1:
            res = snap.resource or "code_search"
            lim = snap.limit or "?"
            print(f"github: [{self._lane}] {res} budget ~{rem}/{lim} requests left", flush=True)
        return data

    def code_search_rate_limited(
        self, query: str, *, page: int, min_interval: float
    ) -> dict[str, Any]:
        self._proactive_throttle(for_search=True)
        wait = min_interval - (time.monotonic() - self._last_code_search)
        if self._last_code_search and wait > 0:
            time.sleep(wait)
        data = self.code_search(query, page=page)
        self._last_code_search = time.monotonic()
        return data

    def fetch_file_bytes(
        self,
        repo_full_name: str,
        path: str,
        *,
        blob_sha: str,
        commit_ref: str = "",
        min_interval: float,
    ) -> bytes:
        """Download file content from a code-search hit.

        Search ``sha`` is a blob OID — not valid as Contents/raw ``ref``.
        """
        wait = min_interval - (time.monotonic() - self._last_download)
        if self._last_download and wait > 0:
            time.sleep(wait)

        norm_path = path.lstrip("/")
        errors: list[str] = []

        if blob_sha:
            try:
                url = f"{API_ROOT}/repos/{repo_full_name}/git/blobs/{blob_sha}"
                data, _hdr = self._request(
                    url,
                    accept="application/vnd.github.raw",
                    min_interval=0.0,
                    last_ts_attr="_last_download",
                    raw_bytes=True,
                )
                if isinstance(data, bytes) and data:
                    return data
            except GitHubFetchError as exc:
                if exc.gone:
                    errors.append(f"git/blobs/{blob_sha[:8]}: gone")
                else:
                    raise
            except RuntimeError as exc:
                errors.append(str(exc))

        ref = commit_ref or ""
        if self._use_raw_cdn and ref:
            cdn_url = f"{RAW_CDN_ROOT}/{repo_full_name}/{ref}/{norm_path}"
            try:
                data, _hdr = self._request(
                    cdn_url,
                    accept="application/octet-stream",
                    min_interval=0.0,
                    last_ts_attr="_last_download",
                    raw_bytes=True,
                )
                if isinstance(data, bytes) and data:
                    return data
            except GitHubFetchError as exc:
                if exc.gone:
                    errors.append("raw CDN: gone")
                else:
                    raise
            except RuntimeError as exc:
                errors.append(str(exc))

        if ref:
            encoded_path = urllib.parse.quote(norm_path, safe="")
            api_url = (
                f"{API_ROOT}/repos/{repo_full_name}/contents/{encoded_path}"
                f"?ref={urllib.parse.quote(ref, safe='')}"
            )
            try:
                data, _hdr = self._request(
                    api_url,
                    accept="application/vnd.github.raw",
                    min_interval=0.0,
                    last_ts_attr="_last_download",
                    raw_bytes=True,
                )
                if isinstance(data, bytes) and data:
                    return data
                if isinstance(data, dict) and data.get("encoding") == "base64":
                    return base64.b64decode(data["content"])
            except GitHubFetchError as exc:
                if exc.gone:
                    errors.append(f"contents?ref={ref[:8]}: gone")
                else:
                    raise
            except RuntimeError as exc:
                errors.append(str(exc))

        detail = "; ".join(errors) if errors else "no fetch strategy succeeded"
        raise GitHubFetchError(
            f"Could not fetch {repo_full_name}/{path} ({detail})",
            gone=bool(errors),
        )

    def fetch_raw_file(
        self,
        repo_full_name: str,
        path: str,
        *,
        sha: str,
        min_interval: float,
        commit_ref: str = "",
    ) -> bytes:
        """Backward-compatible wrapper — prefer ``fetch_file_bytes``."""
        return self.fetch_file_bytes(
            repo_full_name,
            path,
            blob_sha=sha,
            commit_ref=commit_ref,
            min_interval=min_interval,
        )


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"seen": {}, "rejected": {}, "queries": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"seen": {}, "rejected": {}, "queries": {}}
    doc.setdefault("seen", {})
    doc.setdefault("rejected", {})
    doc.setdefault("queries", {})
    return doc


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seen_key(repo: str, file_path: str) -> str:
    return f"{repo}:{file_path}"


def _cache_hit(entry: dict[str, Any] | None, sha: str) -> bool:
    return bool(entry and entry.get("sha") == sha)


def should_skip_path(path: str, cfg: HarvestConfig) -> bool:
    lower = path.lower().replace("\\", "/")
    return any(x.lower() in lower for x in cfg.exclude_path_substrings)


def should_skip_repo(repo: str, cfg: HarvestConfig) -> bool:
    return any(repo.startswith(prefix) for prefix in cfg.exclude_repo_prefixes)


def _normalize_harvest_path(path: str) -> str:
    """Slash-normalize and strip leading ./ for stable regex matching."""
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.lstrip("/")


def _anchored_path_regex(pattern: str) -> re.Pattern[str]:
    """Match path regex from repo-root or after / (patterns with ^ left alone)."""
    if re.search(r"(?:\(\?[^)]*\))?\^", pattern) or pattern.startswith("(?:^"):
        return re.compile(pattern)
    flag_m = re.match(r"(\(\?[a-zA-Z]+\))", pattern)
    if flag_m:
        flags = flag_m.group(1)
        rest = pattern[len(flags) :]
        if rest.startswith("^") or rest.startswith("(?:^"):
            return re.compile(pattern)
        return re.compile(f"{flags}(?:^|/){rest}")
    return re.compile(f"(?:^|/){pattern}")


def _path_regex_search(path: str, pattern: str, *, anchored: bool) -> bool:
    norm = _normalize_harvest_path(path)
    if anchored:
        return _anchored_path_regex(pattern).search(norm) is not None
    return re.search(pattern, norm) is not None


def _path_matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    return any(_path_regex_search(path, p, anchored=True) for p in patterns)


def _path_rejected_by_any(path: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    return any(_path_regex_search(path, p, anchored=False) for p in patterns)


def should_accept_path(path: str, query_spec: dict[str, Any], cfg: HarvestConfig) -> bool:
    """Path allowlist + global/per-query regex — reject before download."""
    norm = _normalize_harvest_path(path)
    if should_skip_path(norm, cfg):
        return False
    if _path_rejected_by_any(norm, cfg.exclude_path_regex):
        return False
    per_ex = tuple(query_spec.get("exclude_path_regex") or ())
    if _path_rejected_by_any(norm, per_ex):
        return False

    required = query_spec.get("require_path_substrings") or ()
    if required:
        lower = norm.lower()
        if not all(req.lower() in lower for req in required):
            return False

    req_re = tuple(query_spec.get("require_path_regex") or ())
    if req_re and not _path_matches_any(norm, req_re):
        return False

    return True


_OPENHANDS_DETECT_PATH_REGEX = (
    r"(?i)(?:^|/)\.openhands-state/sessions/[^/]+/events/.+\.json$",
    r"(?i)(?:^|/)conversations/.+/events/.+\.json$",
)


def detect_harness(path: str, hint: str | None = None) -> str:
    if hint and hint != "generic":
        return hint
    p = path.lower().replace("\\", "/")
    if "agent-transcripts" in p and ".cursor/projects" in p:
        return "cursor"
    if "/chatsessions/" in p and p.endswith(".jsonl"):
        return "copilot_vscode"
    if "github.copilot-chat" in p:
        return "copilot_vscode"
    if "events.jsonl" in p and ".copilot/" in p:
        return "copilot"
    if "claude-session.jsonl" in p:
        return "claude_code"
    if ".codex/sessions" in p:
        return "codex"
    if ".pi/agent/sessions" in p:
        return "pi"
    if ".claude/projects" in p:
        return "claude_code"
    if "opencode/storage/session" in p or "opencode/storage/message" in p or "opencode/storage/part" in p or ".local/share/opencode" in p:
        return "opencode"
    if ".gemini/tmp" in p and "/chats/" in p:
        return "gemini_cli"
    if "antigravity-cache" in p:
        return "antigravity"
    if "trae-cache" in p:
        return "trae"
    if (".qwen/tmp" in p or ".qwen/projects" in p) and "/chats/" in p:
        return "qwen_cli"
    if ".kimi/sessions" in p:
        return "kimi"
    if ".factory/" in p:
        return "factory"
    if ".clawdbot/" in p:
        return "openclaw"
    if ".openclaw/" in p:
        return "openclaw"
    if _path_matches_any(path, _OPENHANDS_DETECT_PATH_REGEX):
        return "openhands"
    if "/amp/threads/" in p or ".local/share/amp/threads/" in p:
        return "amp"
    if ".continue/sessions" in p or ".continue/projects/" in p:
        return "continue"
    if "saoudrizwan.claude-dev" in p or ".cline/data" in p:
        return "cline"
    if "rooveterinaryinc.roo-cline" in p:
        return "roo_code"
    if ".watchfire/logs" in p:
        return "watchfire"
    if ".mux/sessions" in p:
        return "mux"
    if ".kiro/sessions" in p:
        return "kiro"
    if "aider.chat.history" in p:
        return "aider"
    return hint or "generic"


def _line_looks_like_chat(obj: dict[str, Any], harness: str) -> bool:
    if harness == "cursor":
        role = obj.get("role")
        kind = obj.get("type")
        ok_role = role in ("user", "assistant") or kind in ("human", "assistant", "user")
        if not ok_role:
            return False
        message = obj.get("message")
        content: Any = None
        if isinstance(message, dict):
            content = message.get("content")
        if content is None:
            content = obj.get("content")
        return isinstance(content, (str, list))
    if harness == "codex":
        top = obj.get("type")
        payload = obj.get("payload") or {}
        if not isinstance(payload, dict):
            return False
        if top == "event_msg":
            msg = payload.get("message")
            return (
                payload.get("type") == "user_message"
                and isinstance(msg, str)
                and bool(msg.strip())
            )
        if top != "response_item":
            return False
        return (
            payload.get("type") == "message"
            and payload.get("role") in ("user", "developer", "assistant")
        )
    if harness == "copilot":
        from llm_dataprep.copilot_cli import ASSISTANT_TYPES, USER_TYPES

        etype = obj.get("type")
        return etype in USER_TYPES or etype in ASSISTANT_TYPES
    if harness == "copilot_vscode":
        kind = obj.get("kind")
        if kind == 0:
            v = obj.get("v") or {}
            reqs = v.get("requests")
            return isinstance(reqs, list) and any(isinstance(r, dict) for r in reqs)
        if kind == 2:
            keys = obj.get("k") or []
            val = obj.get("v")
            return (
                keys == ["requests"]
                and isinstance(val, list)
                and any(isinstance(r, dict) for r in val)
            )
        return False
    if harness == "kimi":
        role = obj.get("role")
        text = obj.get("content") or obj.get("text")
        return role in ("user", "assistant", "tool") and isinstance(text, (str, list))
    if harness == "pi":
        if obj.get("type") == "custom_message":
            if not obj.get("display", True):
                return False
            content = obj.get("content")
            if isinstance(content, str):
                return bool(content.strip())
            return isinstance(content, list) and bool(content)
        if obj.get("type") != "message":
            return False
        msg = obj.get("message") or {}
        return msg.get("role") in ("user", "assistant")
    if harness == "claude_code":
        t = obj.get("type")
        if t in ("user", "assistant", "human"):
            return bool(obj.get("message"))
        message = obj.get("message")
        if isinstance(message, dict):
            return message.get("role") in ("user", "assistant")
        return False
    if harness in ("gemini_cli", "qwen_cli"):
        if isinstance(obj.get("messages"), list) and obj["messages"]:
            return any(
                isinstance(m, dict)
                and (m.get("type") in ("user", "gemini", "assistant", "model") or m.get("role"))
                for m in obj["messages"]
            )
        return obj.get("type") in ("user", "gemini", "assistant", "model")
    if harness in ("antigravity", "trae"):
        role, text = _role_text(obj)
        return role is not None and bool(text)
    if harness == "openhands":
        role, text = _text_from_event(obj)
        return role in ("user", "assistant") and bool(text.strip())
    if harness == "continue":
        role, text = _text_from_history_item(obj)
        return role in ("user", "assistant") and bool(text.strip())
    if harness == "opencode":
        role, text = role_and_text_from_opencode(obj)
        return role in ("user", "assistant") and bool(text.strip())
    if harness == "amp":
        role = obj.get("role")
        if role not in ("user", "assistant"):
            return False
        return bool(_text_from_amp_message(obj))
    if harness == "openclaw":
        if obj.get("type") not in ("message", "custom_message"):
            return False
        msg = obj.get("message") or obj
        return msg.get("role") in ("user", "assistant")
    if harness == "factory":
        if isinstance(obj.get("messages"), list):
            for msg in obj.get("messages") or []:
                if isinstance(msg, dict) and _line_looks_like_chat(msg, "factory"):
                    return True
            return False
        role = obj.get("role") or obj.get("type")
        if role in ("human", "Human"):
            role = "user"
        if role not in ("user", "assistant"):
            return False
        text = obj.get("content") or obj.get("text")
        return isinstance(text, (str, list)) and bool(text)
    if harness in ("watchfire", "generic"):
        role = obj.get("role") or obj.get("type")
        if role in ("human", "Human"):
            role = "user"
        if role not in ("user", "assistant", "tool", "developer"):
            return False
        for key in GENERIC_TEXT_KEYS:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return True
        message = obj.get("message")
        return isinstance(message, dict) and isinstance(message.get("content"), (str, list))
    role = obj.get("role")
    if role not in ("user", "assistant", "tool", "developer"):
        return False
    for key in GENERIC_TEXT_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return True
    message = obj.get("message")
    return isinstance(message, dict) and isinstance(message.get("content"), (str, list))


def looks_like_chat_blob(
    text: str,
    harness: str,
    *,
    min_hits: int = 2,
    sample_lines: int = 40,
) -> bool:
    """Fast pre-sniff for tests/docs — harvest ingest uses parse_blob_text as gate."""
    if harness == "aider":
        headers = re.findall(r"(?m)^####\s+(user|assistant)\s*$", text[:50_000], re.I)
        if len(headers) >= min_hits:
            return True
        return bool(headers) and "# aider chat started" in text[:5000].lower()

    effective_min = min_hits
    non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(non_empty) < min_hits:
        effective_min = 1

    stripped = text.lstrip()
    if stripped.startswith("[") and harness in ("cline", "roo_code"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, list):
            return False
        from llm_dataprep.cline_tasks import _messages_from_api_history, _messages_from_ui

        for parser in (_messages_from_api_history, _messages_from_ui):
            hits = sum(1 for _ in parser(data))
            if hits >= min_hits:
                return True
        return False

    if stripped.startswith("{") and harness in (
        "continue",
        "opencode",
        "cline",
        "roo_code",
        "amp",
        "openhands",
        "kiro",
        "trae",
        "qwen_cli",
    ):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):

                def _chat_message_count(items: list[Any]) -> int:
                    return sum(
                        1
                        for item in items
                        if isinstance(item, dict) and _line_looks_like_chat(item, harness)
                    )

                messages = obj.get("messages")
                if isinstance(messages, list) and _chat_message_count(messages) >= min_hits:
                    return True
                if _line_looks_like_chat(obj, harness):
                    return True
                history = obj.get("history") or obj.get("chatHistory")
                if isinstance(history, list) and _chat_message_count(history) >= min_hits:
                    return True
            return False

    lines = non_empty[:sample_lines]
    if not lines:
        return False
    hits = 0
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if obj.get("type") in ("summary", "file-history", "progress", "system"):
                continue
            if _line_looks_like_chat(obj, harness):
                hits += 1
    return hits >= effective_min


def _provenance(hit: CodeHit) -> dict[str, str]:
    return {
        "source": "github_public",
        "github_repo": hit.repo_full_name,
        "github_path": hit.path,
        "github_sha": hit.sha,
        "github_url": hit.html_url,
        "github_query_id": hit.query_id,
    }


def _codex_session_id_from_path(path: str) -> str:
    stem = Path(path).stem
    m = _ROLLOUT_TRAILING_UUID_RE.search(stem)
    if m:
        return m.group(1)
    if stem.startswith("rollout-"):
        return stem.removeprefix("rollout-")[:36]
    return stem


def _text_from_codex_user_message(payload: dict[str, Any]) -> str:
    from llm_dataprep.codex_sessions import USER_BLOCK_RE

    text = (payload.get("message") or "").strip()
    if not text:
        return ""
    m = USER_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text


def _parse_codex_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.codex_sessions import SKIP_TYPES, _text_from_codex_content

    session_id = _codex_session_id_from_path(hit.path)
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        top = obj.get("type")
        if top in SKIP_TYPES and top != "event_msg":
            continue
        payload = obj.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        role: str | None = None
        text = ""
        if top == "event_msg":
            if payload.get("type") != "user_message":
                continue
            role = "user"
            text = _text_from_codex_user_message(payload)
        elif top == "response_item":
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "developer", "assistant"):
                continue
            content = payload.get("content")
            if not isinstance(content, list):
                continue
            text = _text_from_codex_content(content)
        else:
            continue
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "codex",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "record_type": top,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _text_from_vscode_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, list):
        parts: list[str] = []
        for item in response:
            if isinstance(item, dict):
                val = item.get("value") or item.get("text") or ""
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        return "\n".join(parts)
    if isinstance(response, dict):
        val = response.get("value") or response.get("text") or ""
        return val.strip() if isinstance(val, str) else ""
    return ""


def _emit_vscode_request_records(
    requests: list[Any],
    *,
    hit: CodeHit,
    session_id: str,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    record_no = 0
    for req in requests:
        if not isinstance(req, dict):
            continue
        message = req.get("message") or {}
        user_text = message.get("text", "") if isinstance(message, dict) else str(message)
        user_text = user_text.strip() if isinstance(user_text, str) else ""
        if user_text:
            record_no += 1
            rec = {
                "harness": "copilot_vscode",
                "session_id": session_id,
                "source_path": f"github:{hit.repo_full_name}/{hit.path}",
                "line_no": record_no,
                "role": "user",
                "text": user_text,
                "ingested_at": ingested_at,
            }
            rec.update(_provenance(hit))
            yield rec
        assistant_text = _text_from_vscode_response(req.get("response"))
        if assistant_text and len(assistant_text) <= 200_000:
            record_no += 1
            rec = {
                "harness": "copilot_vscode",
                "session_id": session_id,
                "source_path": f"github:{hit.repo_full_name}/{hit.path}",
                "line_no": record_no,
                "role": "assistant",
                "text": assistant_text,
                "ingested_at": ingested_at,
            }
            rec.update(_provenance(hit))
            yield rec


def _parse_vscode_copilot_jsonl(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    session_id = Path(hit.path).stem or hit.sha[:12]
    requests: list[Any] = []
    for line in lines[:max_lines]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if kind == 0:
            v = entry.get("v") or {}
            if isinstance(v, dict):
                session_id = v.get("sessionId") or session_id
                init_reqs = v.get("requests")
                if isinstance(init_reqs, list):
                    requests.extend(init_reqs)
        elif kind == 2:
            keys = entry.get("k") or []
            val = entry.get("v")
            if keys == ["requests"] and isinstance(val, list):
                requests.extend(val)
    yield from _emit_vscode_request_records(
        requests,
        hit=hit,
        session_id=session_id,
        ingested_at=ingested_at,
    )


def _parse_vscode_copilot_blob(
    text: str,
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("requests"), list):
            session_id = obj.get("sessionId") or Path(hit.path).stem or hit.sha[:12]
            yield from _emit_vscode_request_records(
                obj["requests"],
                hit=hit,
                session_id=session_id,
                ingested_at=ingested_at,
            )
            return
    yield from _parse_vscode_copilot_jsonl(
        text.splitlines(),
        hit=hit,
        max_lines=max_lines,
        ingested_at=ingested_at,
    )


def _parse_copilot_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.copilot_cli import ASSISTANT_TYPES, USER_TYPES

    session_id = Path(hit.path).parent.name or hit.sha[:12]
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("ephemeral"):
            continue
        etype = ev.get("type")
        data = ev.get("data") or {}
        role = None
        if etype in USER_TYPES:
            role = "user"
        elif etype in ASSISTANT_TYPES:
            role = "assistant"
        if not role:
            continue
        text = (data.get("content") or "").strip()
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "copilot",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "event_type": etype,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_kimi_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.kimi_sessions import _text_from_kimi_message

    session_id = Path(hit.path).parent.name or hit.sha[:12]
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("role")
        if role == "_system_prompt":
            continue
        if role not in ("user", "assistant", "tool"):
            continue
        text = _text_from_kimi_message(obj)
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "kimi",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_openclaw_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.openclaw_sessions import _message_text

    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") not in ("message", "custom_message"):
            continue
        msg = obj.get("message") or obj
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _message_text(msg)
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "openclaw",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_factory_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.factory_droid import _role_text

    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "messages" in obj:
            for msg in obj.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                role, text = _role_text(msg)
                if not role or not text or len(text) > 200_000:
                    continue
                rec = {
                    "harness": "factory",
                    "session_id": session_id,
                    "source_path": f"github:{hit.repo_full_name}/{hit.path}",
                    "line_no": i,
                    "role": role,
                    "text": text,
                    "ingested_at": ingested_at,
                }
                rec.update(_provenance(hit))
                yield rec
            continue
        role, text = _role_text(obj)
        if not role or not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "factory",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_generic_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    harness: str,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.factory_droid import _role_text as factory_role_text
    from llm_dataprep.openclaw_sessions import _message_text

    session_id = Path(hit.path).stem
    effective_harness = harness

    def _emit(role: str, text: str, line_no: int, *, out_harness: str | None = None) -> Iterator[dict[str, Any]]:
        if not text or len(text) > 200_000:
            return
        rec = {
            "harness": out_harness or effective_harness,
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": line_no,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec

    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        if (
            harness in ("openclaw", "generic")
            and obj.get("type") in ("message", "custom_message")
        ):
            msg = obj.get("message") or obj
            role = msg.get("role")
            if role in ("user", "assistant"):
                text = _message_text(msg)
                out_h = "openclaw" if harness == "generic" else harness
                yield from _emit(role, text, i, out_harness=out_h)
                continue

        messages = obj.get("messages")
        if isinstance(messages, list) and harness in ("factory", "generic"):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role, text = factory_role_text(msg)
                if role and text:
                    out_h = "factory" if harness in ("generic", "factory") else harness
                    yield from _emit(role, text, i, out_harness=out_h)
            continue

        role = obj.get("role") or obj.get("type")
        if role in ("human", "Human"):
            role = "user"
        if role not in ("user", "assistant", "tool", "developer"):
            continue
        text = ""
        for key in GENERIC_TEXT_KEYS:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
            if isinstance(val, dict):
                inner = val.get("content") or val.get("text")
                if isinstance(inner, str) and inner.strip():
                    text = inner.strip()
                    break
        if not text and isinstance(obj.get("message"), dict):
            msg = obj["message"]
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = _message_text(msg)
        if not text or len(text) > 200_000:
            continue
        yield from _emit(role, text, i)


def _parse_pi_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.pi_sessions import SKIP_TYPES, _text_from_message

    session_id = Path(hit.path).stem.split("_")[-1] if "_" in Path(hit.path).stem else Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") in SKIP_TYPES:
            continue
        if obj.get("type") == "custom_message" and not obj.get("display", True):
            continue
        if obj.get("type") not in ("message", "custom_message"):
            continue
        msg = obj.get("message") if obj.get("type") == "message" else None
        if obj.get("type") == "custom_message":
            role = "user"
            text = obj.get("content")
            if isinstance(text, list):
                text = _text_from_message({"content": text})
            elif not isinstance(text, str):
                text = ""
        else:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _text_from_message(msg)
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "pi",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_claude_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.claude_sessions import _role_and_text

    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") in ("summary", "file-history", "progress", "system", "local-command"):
            continue
        role, text = _role_and_text(obj)
        if not role or not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "claude_code",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _text_from_qwen_message(obj: dict[str, Any]) -> tuple[str | None, str]:
    """Qwen Code CLI ChatRecord / stream-json: message.parts[] or message.content[]."""
    mtype = obj.get("type")
    role = "user" if mtype == "user" else "assistant" if mtype in ("assistant", "model") else None
    if not role:
        return None, ""
    message = obj.get("message") or {}
    if isinstance(message, dict):
        parts = message.get("parts")
        if isinstance(parts, list):
            text = "\n".join(
                (p.get("text") or "") for p in parts if isinstance(p, dict)
            ).strip()
            if text:
                return role, text
    for content in (
        (message.get("content") if isinstance(message, dict) else None),
        obj.get("content"),
    ):
        if isinstance(content, str):
            text = content.strip()
            m = USER_QUERY_RE.search(text)
            text = m.group(1).strip() if m else text
        elif isinstance(content, list):
            text, _ = text_from_content_blocks(content)
        else:
            continue
        if text:
            return role, text
    return None, ""


def _parse_qwen_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        role, text = _text_from_qwen_message(obj)
        if not role or not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "qwen_cli",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_opencode_part_json(
    text: str,
    *,
    hit: CodeHit,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    """OpenCode split storage/part/*.json text blobs."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        return
    if obj.get("type") != "text":
        return
    body = (obj.get("text") or "").strip()
    if not body or len(body) > 200_000:
        return
    session_id = Path(hit.path).parent.name or Path(hit.path).stem
    rec = {
        "harness": "opencode",
        "session_id": session_id,
        "source_path": f"github:{hit.repo_full_name}/{hit.path}",
        "line_no": 1,
        "role": "assistant",
        "text": body,
        "ingested_at": ingested_at,
    }
    rec.update(_provenance(hit))
    yield rec


def _parse_gemini_lines(
    lines: list[str],
    *,
    hit: CodeHit,
    harness: str,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    """JSONL lines — mirrors gemini_cli._parse_jsonl (type-based turns)."""
    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line or line.startswith('{"$set"'):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        pairs: list[tuple[str, str]] = []
        if "messages" in obj and isinstance(obj["messages"], list):
            pairs = list(_messages_from_obj(obj))
        else:
            mtype = obj.get("type")
            if mtype == "user":
                text = (obj.get("content") or "").strip()
                if text:
                    pairs.append(("user", text))
            elif mtype in ("gemini", "assistant", "model"):
                text = (obj.get("content") or obj.get("text") or "").strip()
                if text:
                    pairs.append(("assistant", text))
        for role, text in pairs:
            if not text or len(text) > 200_000:
                continue
            rec = {
                "harness": harness,
                "session_id": session_id,
                "source_path": f"github:{hit.repo_full_name}/{hit.path}",
                "line_no": i,
                "role": role,
                "text": text,
                "ingested_at": ingested_at,
            }
            rec.update(_provenance(hit))
            yield rec


def _parse_gemini_json_blob(
    text: str,
    *,
    hit: CodeHit,
    harness: str,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    """Whole session JSON — mirrors gemini_cli ingest for .json chats."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        return
    session_id = Path(hit.path).stem
    for line_no, (role, text) in enumerate(_messages_from_obj(obj), start=1):
        if not text or len(text) > 200_000:
            continue
        rec = {
            "harness": harness,
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": line_no,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_tokscale_jsonl(
    lines: list[str],
    *,
    hit: CodeHit,
    harness: str,
    max_lines: int,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    """Role-based JSONL — mirrors tokscale_cache._records_from_jsonl."""
    session_id = Path(hit.path).stem
    for i, line in enumerate(lines[:max_lines], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        role, text = _role_text(obj)
        if not role or not text or len(text) > 200_000:
            continue
        rec = {
            "harness": harness,
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_trae_json(
    text: str,
    *,
    hit: CodeHit,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    """Whole session JSON — mirrors tokscale_cache._records_from_json_files."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    session_id = Path(hit.path).stem
    messages: list[Any] = []
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages") or data.get("turns") or []
    if not isinstance(messages, list):
        return
    for i, obj in enumerate(messages, start=1):
        if not isinstance(obj, dict):
            continue
        role, text = _role_text(obj)
        if not role or not text or len(text) > 200_000:
            continue
        rec = {
            "harness": "trae",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_aider_markdown(
    text: str,
    *,
    hit: CodeHit,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    session_id = Path(hit.path).stem
    parts = re.split(r"(?m)^####\s+(user|assistant)\s*$", text)
    if len(parts) < 3:
        return
    line_no = 0
    for idx in range(1, len(parts), 2):
        role = parts[idx].strip().lower()
        body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        if not body or len(body) > 200_000:
            continue
        line_no += 1
        rec = {
            "harness": "aider",
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": line_no,
            "role": role,
            "text": body,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _openhands_session_id(hit: CodeHit) -> str:
    parts = Path(hit.path).parts
    if "events" in parts:
        idx = parts.index("events")
        if idx > 0:
            return parts[idx - 1]
    return Path(hit.path).stem


def _parse_openhands_event_json(
    obj: dict[str, Any],
    *,
    hit: CodeHit,
    ingested_at: str,
    line_no: int = 1,
) -> Iterator[dict[str, Any]]:
    role, body = _text_from_event(obj)
    if role not in ("user", "assistant") or not body.strip():
        return
    if len(body) > 200_000:
        return
    rec = {
        "harness": "openhands",
        "session_id": _openhands_session_id(hit),
        "source_path": f"github:{hit.repo_full_name}/{hit.path}",
        "line_no": line_no,
        "role": role,
        "text": body.strip(),
        "ingested_at": ingested_at,
    }
    rec.update(_provenance(hit))
    yield rec


def _parse_cline_task_json(
    text: str,
    *,
    hit: CodeHit,
    harness: str,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    from llm_dataprep.cline_tasks import _messages_from_api_history, _messages_from_ui

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(data, list):
        return

    name = Path(hit.path).name.lower()
    if name == "ui_messages.json":
        parser = _messages_from_ui
    else:
        parser = _messages_from_api_history

    session_id = Path(hit.path).parent.name or Path(hit.path).stem
    for i, (role, body) in enumerate(parser(data), start=1):
        if len(body) > 200_000:
            continue
        rec = {
            "harness": harness,
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": i,
            "role": role,
            "text": body,
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec


def _parse_whole_json_blob(
    text: str,
    *,
    hit: CodeHit,
    harness: str,
    ingested_at: str,
) -> Iterator[dict[str, Any]]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        return
    if harness == "openhands" and obj.get("source") in ("user", "agent", "assistant"):
        yield from _parse_openhands_event_json(obj, hit=hit, ingested_at=ingested_at)
        return
    session_id = Path(hit.path).stem
    if harness == "amp" and obj.get("id"):
        session_id = str(obj["id"])

    def emit(role: str | None, body: str, line_no: int) -> Iterator[dict[str, Any]]:
        if role not in ("user", "assistant") or not body.strip():
            return
        if len(body) > 200_000:
            return
        rec = {
            "harness": harness,
            "session_id": session_id,
            "source_path": f"github:{hit.repo_full_name}/{hit.path}",
            "line_no": line_no,
            "role": role,
            "text": body.strip(),
            "ingested_at": ingested_at,
        }
        rec.update(_provenance(hit))
        yield rec

    if harness == "opencode":
        role, text_val = role_and_text_from_opencode(obj)
        if role in ("user", "assistant") and text_val:
            yield from emit(role, text_val, 1)
    elif obj.get("role") in ("user", "assistant"):
        text_val = obj.get("content") or obj.get("text") or ""
        if isinstance(text_val, str) and text_val.strip():
            yield from emit(obj["role"], text_val, 1)
            return

    messages = obj.get("messages") or obj.get("history") or obj.get("chatHistory") or []
    if isinstance(messages, list):
        for i, msg in enumerate(messages, start=1):
            if not isinstance(msg, dict):
                continue
            if harness == "continue":
                role, content = _text_from_history_item(msg)
                yield from emit(role, content, i)
                continue
            if harness == "opencode":
                role, content = role_and_text_from_opencode(msg)
                yield from emit(role, content, i)
                continue
            if harness == "amp":
                role = msg.get("role")
                if role in ("user", "assistant"):
                    body = _text_from_amp_message(msg)
                    yield from emit(role, body, i)
                continue
            role = msg.get("role") or msg.get("type")
            if role in ("human", "Human"):
                role = "user"
            if role in ("gemini", "model"):
                role = "assistant"
            content = msg.get("content") or msg.get("text") or msg.get("message") or ""
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if isinstance(content, str):
                yield from emit(str(role) if role else None, content, i)

    if harness == "opencode" and obj.get("role") in ("user", "assistant"):
        return


def parse_blob_text(
    hit: CodeHit,
    text: str,
    *,
    harness_hint: str | None,
    max_lines: int,
) -> Iterator[dict[str, Any]]:
    ingested_at = datetime.now(timezone.utc).isoformat()
    harness = detect_harness(hit.path, harness_hint)
    lines = text.splitlines()

    if harness == "cursor":
        sid = session_id_from_path(Path(hit.path))
        for i, line in enumerate(lines[:max_lines], start=1):
            rec = parse_cursor_line(
                line,
                session_id=sid,
                source_path=f"github:{hit.repo_full_name}/{hit.path}",
                line_no=i,
            )
            if rec is None:
                continue
            d = rec.to_dict()
            d["source"] = "github_public"
            d.update(_provenance(hit))
            yield d
        return

    if harness == "codex":
        yield from _parse_codex_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "copilot":
        yield from _parse_copilot_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "copilot_vscode":
        yield from _parse_vscode_copilot_blob(
            text,
            hit=hit,
            max_lines=max_lines,
            ingested_at=ingested_at,
        )
        return

    if harness == "kimi":
        yield from _parse_kimi_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "openclaw":
        yield from _parse_openclaw_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "factory":
        yield from _parse_factory_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "pi":
        yield from _parse_pi_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "claude_code":
        yield from _parse_claude_lines(lines, hit=hit, max_lines=max_lines, ingested_at=ingested_at)
        return

    if harness == "aider":
        yield from _parse_aider_markdown(text, hit=hit, ingested_at=ingested_at)
        return

    if harness in ("cline", "roo_code"):
        if hit.path.lower().endswith(".json"):
            yield from _parse_cline_task_json(
                text, hit=hit, harness=harness, ingested_at=ingested_at
            )
            return

    if harness == "qwen_cli":
        if hit.path.lower().endswith(".json"):
            yield from _parse_gemini_json_blob(
                text, hit=hit, harness=harness, ingested_at=ingested_at
            )
        else:
            yield from _parse_qwen_lines(
                lines,
                hit=hit,
                max_lines=max_lines,
                ingested_at=ingested_at,
            )
        return

    if harness == "gemini_cli":
        if hit.path.lower().endswith(".json"):
            yield from _parse_gemini_json_blob(
                text, hit=hit, harness=harness, ingested_at=ingested_at
            )
        else:
            yield from _parse_gemini_lines(
                lines,
                hit=hit,
                harness=harness,
                max_lines=max_lines,
                ingested_at=ingested_at,
            )
        return

    if harness == "opencode" and "/storage/part/" in hit.path.replace("\\", "/").lower():
        yield from _parse_opencode_part_json(text, hit=hit, ingested_at=ingested_at)
        return

    if harness in ("continue", "opencode", "amp", "openhands") and hit.path.lower().endswith(".json"):
        yield from _parse_whole_json_blob(
            text, hit=hit, harness=harness, ingested_at=ingested_at
        )
        return

    if harness == "kiro":
        if hit.path.lower().endswith(".json"):
            yield from _parse_whole_json_blob(
                text, hit=hit, harness=harness, ingested_at=ingested_at
            )
        else:
            yield from _parse_generic_lines(
                lines,
                hit=hit,
                harness=harness,
                max_lines=max_lines,
                ingested_at=ingested_at,
            )
        return

    if harness == "trae":
        yield from _parse_trae_json(text, hit=hit, ingested_at=ingested_at)
        return

    if harness == "antigravity":
        yield from _parse_tokscale_jsonl(
            lines,
            hit=hit,
            harness="antigravity",
            max_lines=max_lines,
            ingested_at=ingested_at,
        )
        return

    yield from _parse_generic_lines(
        lines,
        hit=hit,
        harness=harness,
        max_lines=max_lines,
        ingested_at=ingested_at,
    )


def pick_download_lane_for_hit(lanes: list[HarvestLane], auth_lane: str) -> HarvestLane:
    """Prefer dedicated App download lane; else same lane that ran the search."""
    for lane in lanes:
        if lane.name == "app" and not lane.search:
            return lane
    if auth_lane:
        return lane_by_name(lanes, auth_lane)
    return lanes[0]


def iter_code_search_hits(
    client: GitHubClient | None,
    query_spec: dict[str, Any],
    cfg: HarvestConfig,
    *,
    lanes: list[HarvestLane] | None = None,
    search_request_counter: list[int] | None = None,
    max_pages: int | None = None,
    start_page: int = 1,
    query_progress: dict[str, Any] | None = None,
) -> Iterator[CodeHit]:
    q = query_spec["q"]
    qid = query_spec.get("id") or q[:40]
    pages = max_pages if max_pages is not None else int(
        query_spec.get("max_pages", cfg.default_max_pages)
    )
    start_page = max(1, min(start_page, pages))
    last_page = start_page - 1
    search_requests = 0
    active_lanes = search_lanes(lanes) if lanes else []
    alternate = bool(
        search_request_counter is not None and len(active_lanes) > 1
    )
    if not alternate and client is None:
        raise ValueError("client required when not alternating search lanes")

    for page in range(start_page, pages + 1):
        lane_name = ""
        if alternate:
            lane = pick_search_lane(active_lanes, search_request_counter[0])  # type: ignore[index]
            search_request_counter[0] += 1
            active_client = lane.client
            lane_name = lane.name
            print(f"  search page={page} lane={lane_name}", flush=True)
        else:
            active_client = client  # type: ignore[assignment]
            lane_name = active_client._lane

        data = active_client.code_search_rate_limited(
            q, page=page, min_interval=cfg.code_search_min_interval_s
        )
        search_requests += 1
        items = data.get("items") or []
        if not items:
            break
        last_page = page
        for item in items:
            repo = item.get("repository") or {}
            full_name = repo.get("full_name") or ""
            path = item.get("path") or ""
            sha = item.get("sha") or ""
            html_url = item.get("html_url") or ""
            if not full_name or not path:
                continue
            yield CodeHit(
                repo_full_name=full_name,
                path=path,
                sha=sha,
                html_url=html_url,
                query_id=qid,
                commit_ref=commit_ref_from_html_url(html_url),
                auth_lane=lane_name,
            )
        if len(items) < 100:
            break
    if query_progress is not None:
        query_progress["last_page"] = last_page
        query_progress["search_requests"] = search_requests


def _ingest_hit_text(
    hit: CodeHit,
    text: str,
    *,
    qspec: dict[str, Any],
    qid: str,
    key: str,
    cfg: HarvestConfig,
    cache: HarvestCache,
    stats: dict[str, int],
    record_buf: list[dict[str, Any]],
    flush_records: Any,
) -> None:
    n_recs = 0
    for rec in parse_blob_text(
        hit,
        text,
        harness_hint=qspec.get("harness_hint"),
        max_lines=cfg.max_lines_per_file,
    ):
        record_buf.append(rec)
        n_recs += 1
        if len(record_buf) >= 500:
            flush_records()

    if n_recs == 0:
        print(f"  skip empty parse ({detect_harness(hit.path, qspec.get('harness_hint'))})", flush=True)
        stats["files_skipped"] += 1
        stats["files_rejected_content"] += 1
        cache.set_rejected(
            key,
            {
                "sha": hit.sha,
                "reason": "empty_parse",
                "at": datetime.now(timezone.utc).isoformat(),
                "query_id": qid,
            },
        )
        return

    stats["files_fetched"] += 1
    stats["records"] += n_recs
    cache.set_seen(
        key,
        {
            "sha": hit.sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "records": n_recs,
            "query_id": qid,
        },
    )
    flush_records()


def _flush_pending_downloads(
    pending: list[tuple[CodeHit, str, dict[str, Any], str, str]],
    *,
    lanes: list[HarvestLane],
    cfg: HarvestConfig,
    cache: HarvestCache,
    stats: dict[str, int],
    record_buf: list[dict[str, Any]],
    flush_records: Any,
    dry_run: bool,
) -> None:
    if not pending:
        return

    by_lane: dict[str, list[tuple[CodeHit, str, dict[str, Any], str, str]]] = {}
    for item in pending:
        auth = item[4] or lanes[0].name
        by_lane.setdefault(auth, []).append(item)
    pending.clear()

    for auth_lane, batch in by_lane.items():
        dl_lane = pick_download_lane_for_hit(lanes, auth_lane)
        _flush_pending_downloads_batch(
            batch,
            client=dl_lane.client,
            gql=dl_lane.gql,
            cfg=cfg,
            cache=cache,
            stats=stats,
            record_buf=record_buf,
            flush_records=flush_records,
            dry_run=dry_run,
        )


def _flush_pending_downloads_batch(
    pending: list[tuple[CodeHit, str, dict[str, Any], str, str]],
    *,
    client: GitHubClient,
    gql: GraphQLBlobFetcher | None,
    cfg: HarvestConfig,
    cache: HarvestCache,
    stats: dict[str, int],
    record_buf: list[dict[str, Any]],
    flush_records: Any,
    dry_run: bool,
) -> None:
    if not pending:
        return

    if dry_run:
        for hit, key, _qspec, qid, _auth in pending:
            cache.set_seen(key, {"sha": hit.sha, "dry_run": True, "query_id": qid})
            stats["files_fetched"] += 1
        return

    reqs = [
        BlobRequest(
            hit.repo_full_name,
            hit.path,
            hit.sha,
            hit.commit_ref,
            key,
        )
        for hit, key, _qspec, _qid, _auth in pending
    ]
    blobs: dict[str, bytes] = {}
    if cfg.download_mode in ("hybrid", "graphql") and gql is not None:
        try:
            blobs = gql.fetch_batch(
                reqs,
                max_files_per_repo=cfg.graphql_files_per_repo,
                max_repos_per_query=cfg.graphql_max_repos,
            )
            stats["graphql_blobs"] = stats.get("graphql_blobs", 0) + len(blobs)
        except RuntimeError as exc:
            print(f"github: GraphQL batch failed — REST/raw fallback ({exc})", flush=True)

    for hit, key, qspec, qid, _auth in pending:
        blob = blobs.get(key)
        if blob is None and cfg.download_mode != "graphql":
            try:
                blob = client.fetch_file_bytes(
                    hit.repo_full_name,
                    hit.path,
                    blob_sha=hit.sha,
                    commit_ref=hit.commit_ref,
                    min_interval=cfg.download_min_interval_s,
                )
                stats["rest_blobs"] = stats.get("rest_blobs", 0) + 1
            except GitHubFetchError as exc:
                if exc.gone:
                    print(f"  skip gone: {hit.path}", flush=True)
                    cache.set_rejected(
                        key,
                        {
                            "sha": hit.sha,
                            "reason": "gone",
                            "at": datetime.now(timezone.utc).isoformat(),
                            "query_id": qid,
                        },
                    )
                    stats["files_rejected_gone"] = stats.get("files_rejected_gone", 0) + 1
                else:
                    print(f"  skip fetch error: {exc}", flush=True)
                stats["files_skipped"] += 1
                continue
            except (RuntimeError, OSError, urllib.error.URLError) as exc:
                print(f"  skip fetch error: {exc}", flush=True)
                stats["files_skipped"] += 1
                continue
        elif blob is None and cfg.download_mode == "graphql" and hit.sha:
            try:
                blob = client.fetch_file_bytes(
                    hit.repo_full_name,
                    hit.path,
                    blob_sha=hit.sha,
                    commit_ref=hit.commit_ref,
                    min_interval=cfg.download_min_interval_s,
                )
                stats["rest_blobs"] = stats.get("rest_blobs", 0) + 1
            except GitHubFetchError as exc:
                if exc.gone:
                    print(f"  skip gone: {hit.path}", flush=True)
                    cache.set_rejected(
                        key,
                        {
                            "sha": hit.sha,
                            "reason": "gone",
                            "at": datetime.now(timezone.utc).isoformat(),
                            "query_id": qid,
                        },
                    )
                    stats["files_rejected_gone"] = stats.get("files_rejected_gone", 0) + 1
                else:
                    print(f"  skip fetch error: {exc}", flush=True)
                stats["files_skipped"] += 1
                continue
            except (RuntimeError, OSError, urllib.error.URLError) as exc:
                print(f"  skip fetch error: {exc}", flush=True)
                stats["files_skipped"] += 1
                continue
        if blob is None:
            stats["files_skipped"] += 1
            continue

        if len(blob) > cfg.max_file_bytes:
            print(f"  skip oversized ({len(blob)} bytes)", flush=True)
            stats["files_skipped"] += 1
            continue

        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            text = blob.decode("utf-8", errors="replace")

        if not text.strip():
            stats["files_skipped"] += 1
            continue

        _ingest_hit_text(
            hit,
            text,
            qspec=qspec,
            qid=qid,
            key=key,
            cfg=cfg,
            cache=cache,
            stats=stats,
            record_buf=record_buf,
            flush_records=flush_records,
        )


def run_harvest(
    cfg: HarvestConfig,
    *,
    query_ids: set[str] | None = None,
    dry_run: bool = False,
    reset_state: bool = False,
) -> dict[str, int]:
    _bootstrap_env()
    pat_token = _github_token()
    lanes = build_harvest_lanes(cfg, pat_token)
    if reset_state and cfg.state_path.is_file():
        cfg.state_path.unlink(missing_ok=True)

    cache = HarvestCache(cfg.state_path, redis_url=cfg.redis_url)
    if reset_state:
        cache.clear_all()
    print(f"github-harvest: cache backend={cache.backend}", flush=True)

    stats = {
        "queries_run": 0,
        "hits_seen": 0,
        "files_fetched": 0,
        "files_skipped": 0,
        "files_cache_skipped": 0,
        "files_rejected_path": 0,
        "files_rejected_content": 0,
        "graphql_blobs": 0,
        "rest_blobs": 0,
        "records": 0,
        "search_requests": 0,
        "lanes": len(lanes),
    }

    out_path = dated_raw_path(cfg.raw_prefix, data_dir() / "raw")
    replace_out = not out_path.is_file()
    record_buf: list[dict[str, Any]] = []
    pending: list[tuple[CodeHit, str, dict[str, Any], str, str]] = []
    search_request_counter = [0]
    dual_search = len(search_lanes(lanes)) > 1

    def flush_pending() -> None:
        _flush_pending_downloads(
            pending,
            lanes=lanes,
            cfg=cfg,
            cache=cache,
            stats=stats,
            record_buf=record_buf,
            flush_records=flush_records,
            dry_run=dry_run,
        )

    def flush_records() -> None:
        nonlocal replace_out
        if not record_buf or dry_run:
            record_buf.clear()
            return
        append_records_buffered(
            out_path,
            iter(record_buf),
            buffer_rows=500,
            replace=replace_out,
        )
        replace_out = False
        record_buf.clear()

    queries = cfg.queries
    if query_ids:
        queries = tuple(q for q in queries if q.get("id") in query_ids)

    for query_index, qspec in enumerate(queries):
        if stats["files_fetched"] >= cfg.max_files_per_run:
            break
        qid = qspec.get("id") or qspec["q"][:40]
        if dual_search:
            print(
                f"github-harvest: query={qid} (search alternates pat/app each page)",
                flush=True,
            )
        else:
            search_lane = pick_search_lane(lanes, query_index)
            print(f"github-harvest: query={qid} lane={search_lane.name}", flush=True)
        stats["queries_run"] += 1

        pages = int(qspec.get("max_pages", cfg.default_max_pages))
        qstate = cache.get_query_state(qid) or {}
        start_page = int(qstate.get("last_page", 0)) + 1
        if start_page > pages:
            print(
                f"github-harvest: query={qid} skipped "
                f"(pages 1-{pages} already scanned; clear state to rescan)",
                flush=True,
            )
            continue
        query_progress: dict[str, Any] = {}

        search_budget_hit = False
        single_lane = pick_search_lane(lanes, query_index) if not dual_search else None
        for hit in iter_code_search_hits(
            single_lane.client if single_lane else None,
            qspec,
            cfg,
            lanes=lanes if dual_search else None,
            search_request_counter=search_request_counter if dual_search else None,
            start_page=start_page,
            query_progress=query_progress,
        ):
            stats["hits_seen"] += 1
            if (
                cfg.max_search_requests_per_run is not None
                and stats["search_requests"] >= cfg.max_search_requests_per_run
            ):
                search_budget_hit = True
                break
            if stats["files_fetched"] + len(pending) >= cfg.max_files_per_run:
                break

            key = _seen_key(hit.repo_full_name, hit.path)
            if _cache_hit(cache.get_seen(key), hit.sha):
                stats["files_skipped"] += 1
                stats["files_cache_skipped"] += 1
                continue

            if _cache_hit(cache.get_rejected(key), hit.sha):
                stats["files_skipped"] += 1
                stats["files_cache_skipped"] += 1
                continue

            if should_skip_repo(hit.repo_full_name, cfg):
                stats["files_skipped"] += 1
                continue

            if not should_accept_path(hit.path, qspec, cfg):
                stats["files_skipped"] += 1
                stats["files_rejected_path"] += 1
                cache.set_rejected(
                    key,
                    {
                        "sha": hit.sha,
                        "reason": "path",
                        "at": datetime.now(timezone.utc).isoformat(),
                        "query_id": qid,
                    },
                )
                continue

            print(f"  fetch {hit.repo_full_name}/{hit.path}", flush=True)
            pending.append((hit, key, qspec, qid, hit.auth_lane))
            if len(pending) >= cfg.graphql_pending_flush:
                flush_pending()

        flush_pending()

        stats["search_requests"] += int(query_progress.get("search_requests", 0))
        if query_progress.get("last_page"):
            cache.set_query_state(
                qid,
                {
                    "last_page": query_progress["last_page"],
                    "pages": pages,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        if search_budget_hit:
            break

    flush_records()
    if not dry_run:
        cache.flush()

    rl = lanes[0].client.rate_limit_status()
    if rl.remaining is not None:
        stats["rate_limit_remaining"] = rl.remaining

    return stats


def _app_credentials_from_env(*, require_installation: bool = True) -> tuple[str, bytes, int | None]:
    client_id = os.environ.get("GITHUB_APP_CLIENT_ID", "").strip()
    if not client_id:
        raise SystemExit("GITHUB_APP_CLIENT_ID is required")
    from llm_dataprep.github_harvest_app import _load_private_key_pem

    pem = _load_private_key_pem()
    if pem is None:
        raise SystemExit(
            "GitHub App private key required: set GITHUB_APP_PRIVATE_KEY_PATH or GITHUB_APP_PRIVATE_KEY"
        )
    inst_raw = os.environ.get("GITHUB_APP_INSTALLATION_ID", "").strip()
    installation_id: int | None = None
    if inst_raw:
        try:
            installation_id = int(inst_raw)
        except ValueError:
            raise SystemExit(f"GITHUB_APP_INSTALLATION_ID must be integer, got {inst_raw!r}") from None
    if require_installation and installation_id is None:
        raise SystemExit(
            "GITHUB_APP_INSTALLATION_ID is required (or use --resolve-app-installation USER)"
        )
    return client_id, pem, installation_id


def _run_app_installation_helpers(args: argparse.Namespace) -> None:
    _bootstrap_env()
    client_id, pem, _ = _app_credentials_from_env(require_installation=False)
    if args.list_app_installations:
        rows = list_installations(client_id=client_id, private_key_pem=pem)
        for row in rows:
            acct = row.get("account") or {}
            login = acct.get("login") or "?"
            print(f"{row.get('id')}\t{acct.get('type')}\t{login}")
        return
    if args.resolve_app_installation:
        iid = resolve_user_installation_id(
            client_id=client_id,
            private_key_pem=pem,
            username=args.resolve_app_installation,
        )
        print(f"GITHUB_APP_INSTALLATION_ID={iid}")
        return
    raise SystemExit("No app helper action selected")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search GitHub for public agent session JSONL and ingest to data/raw"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to github-harvest.yaml (default: config/github-harvest.yaml)",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Run only these query ids (repeatable)",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Override max_files_per_run")
    parser.add_argument("--dry-run", action="store_true", help="Search only; do not download/write")
    parser.add_argument("--reset-state", action="store_true", help="Ignore prior seen files")
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print registry query ids and exit",
    )
    parser.add_argument(
        "--list-app-installations",
        action="store_true",
        help="List GitHub App installation IDs (requires app env + private key)",
    )
    parser.add_argument(
        "--resolve-app-installation",
        metavar="GITHUB_USER",
        default=None,
        help="Print GITHUB_APP_INSTALLATION_ID for a user account",
    )
    args = parser.parse_args()

    if args.list_app_installations or args.resolve_app_installation:
        _run_app_installation_helpers(args)
        return

    if args.list_queries:
        cfg = load_harvest_config(args.config)
        for q in cfg.queries:
            label = q.get("label") or q.get("id")
            print(f"{q.get('id')}\t{q.get('harness_hint')}\t{label}")
        return

    cfg = load_harvest_config(args.config)
    if args.max_files is not None:
        cfg = replace(cfg, max_files_per_run=max(1, args.max_files))

    query_ids = set(args.queries) if args.queries else None
    stats = run_harvest(
        cfg,
        query_ids=query_ids,
        dry_run=args.dry_run,
        reset_state=args.reset_state,
    )

    print(
        "github-harvest done: "
        f"queries={stats['queries_run']} hits={stats['hits_seen']} "
        f"fetched={stats['files_fetched']} skipped={stats['files_skipped']} "
        f"cache_skipped={stats.get('files_cache_skipped', 0)} "
        f"rejected_path={stats.get('files_rejected_path', 0)} "
        f"rejected_content={stats.get('files_rejected_content', 0)} "
        f"graphql={stats.get('graphql_blobs', 0)} rest={stats.get('rest_blobs', 0)} "
        f"records={stats['records']}"
        + (
            f" api_remaining={stats['rate_limit_remaining']}"
            if stats.get("rate_limit_remaining") is not None
            else ""
        ),
        flush=True,
    )
    if stats["records"] and not args.dry_run:
        out = dated_raw_path(cfg.raw_prefix, data_dir() / "raw")
        print(f"raw → {out}", flush=True)
        print("next: make github-harvest-full  (scan + curate)", flush=True)


if __name__ == "__main__":
    main()
