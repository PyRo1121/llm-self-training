"""GitHub GraphQL batch blob fetch (pairs with REST code search)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

GRAPHQL_URL = "https://api.github.com/graphql"
USER_AGENT = "llm-dataprep-github-harvest/0.1"


@dataclass(frozen=True)
class BlobRequest:
    repo_full_name: str
    path: str
    blob_sha: str
    commit_ref: str
    cache_key: str


def _split_repo(full_name: str) -> tuple[str, str]:
    owner, _, name = full_name.partition("/")
    if not owner or not name:
        raise ValueError(f"Invalid repo: {full_name}")
    return owner, name


def _git_expression(commit_ref: str, path: str) -> str:
    """Git rev-parse expression: {commit}:{path}"""
    clean = path.lstrip("/")
    if re.search(r"[\[\]~^:?*\\]", clean):
        clean = clean.replace('"', '\\"')
        return f'{commit_ref}:"{clean}"'
    return f"{commit_ref}:{clean}"


def _build_repo_fragment(
    alias: str,
    owner: str,
    name: str,
    files: list[tuple[str, BlobRequest]],
) -> str:
    lines = [f"  {alias}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{"]
    for field_alias, req in files:
        if not req.commit_ref:
            continue
        expr = _git_expression(req.commit_ref, req.path)
        lines.append(
            f"    {field_alias}: object(expression: {json.dumps(expr)}) {{"
            f" ... on Blob {{ oid byteSize isBinary isTruncated text }} }}"
        )
    lines.append("  }")
    return "\n".join(lines)


def _header(headers: dict[str, str], name: str) -> str | None:
    return headers.get(name.lower())


def _parse_reset_at(headers: dict[str, str]) -> float | None:
    reset = _header(headers, "x-ratelimit-reset")
    if reset is None:
        return None
    try:
        return float(reset)
    except ValueError:
        return None


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


def _is_rate_limited_errors(errors: list[Any]) -> bool:
    for err in errors:
        if not isinstance(err, dict):
            continue
        if err.get("type") == "RATE_LIMITED":
            return True
        msg = err.get("message")
        if isinstance(msg, str) and "rate limit" in msg.lower():
            return True
    return False


def _should_retry_rate_limit(
    headers: dict[str, str],
    errors: list[Any],
    *,
    data: Any,
) -> bool:
    if _is_rate_limited_errors(errors):
        return True
    if data is None and _header(headers, "x-ratelimit-remaining") == "0":
        return True
    return False


def build_batch_query(groups: list[tuple[str, str, str, list[tuple[str, BlobRequest]]]]) -> str:
    """groups: (repo_alias, owner, name, [(field_alias, req), ...])"""
    parts = ["query FetchBlobs {"]
    for repo_alias, owner, name, files in groups:
        parts.append(_build_repo_fragment(repo_alias, owner, name, files))
    parts.append("}")
    return "\n".join(parts)


class GraphQLBlobFetcher:
    def __init__(
        self,
        token: str | Callable[[], str],
        *,
        max_retries: int = 5,
        on_rate_limit: Callable[[dict[str, str]], None] | None = None,
    ) -> None:
        self._token_provider: Callable[[], str] = (
            (lambda: token) if isinstance(token, str) else token
        )
        self._max_retries = max(1, max_retries)
        self._on_rate_limit = on_rate_limit

    def _post(self, query: str) -> dict[str, Any]:
        payload = json.dumps({"query": query}).encode("utf-8")
        for attempt in range(self._max_retries):
            req = urllib.request.Request(
                GRAPHQL_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._token_provider()}",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            headers: dict[str, str] = {}
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    if self._on_rate_limit:
                        self._on_rate_limit(headers)
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                err_headers = {k.lower(): v for k, v in exc.headers.items()}
                if self._on_rate_limit:
                    self._on_rate_limit(err_headers)
                if exc.code in (403, 429, 502, 503) and attempt + 1 < self._max_retries:
                    exc.read()
                    sleep_s = _backoff_seconds(
                        attempt,
                        retry_after=err_headers.get("retry-after"),
                        reset_at=_parse_reset_at(err_headers),
                    )
                    print(
                        f"github: GraphQL HTTP {exc.code} — backoff {sleep_s:.0f}s "
                        f"(attempt {attempt + 1}/{self._max_retries})",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"GraphQL HTTP {exc.code}: {detail}") from exc

            errors = body.get("errors") or []
            data = body.get("data")
            if _should_retry_rate_limit(headers, errors, data=data):
                if attempt + 1 >= self._max_retries:
                    if errors:
                        raise RuntimeError(f"GraphQL errors: {errors[:3]}")
                    raise RuntimeError("GraphQL rate limited after retries")
                sleep_s = _backoff_seconds(
                    attempt,
                    retry_after=headers.get("retry-after"),
                    reset_at=_parse_reset_at(headers),
                )
                print(
                    f"github: GraphQL rate limited — backoff {sleep_s:.0f}s "
                    f"(attempt {attempt + 1}/{self._max_retries})",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            if data is None:
                if errors:
                    raise RuntimeError(f"GraphQL errors: {errors[:3]}")
                raise RuntimeError("GraphQL response missing data")
            if not isinstance(data, dict):
                raise RuntimeError("GraphQL response data is not an object")
            if errors:
                print(
                    f"github: GraphQL partial errors ({len(errors)}): {errors[:2]}",
                    flush=True,
                )
            return data
        raise RuntimeError("GraphQL request failed after retries")

    def fetch_batch(
        self,
        requests: list[BlobRequest],
        *,
        max_repos_per_query: int = 3,
        max_files_per_repo: int = 8,
    ) -> dict[str, bytes]:
        """Return cache_key → utf-8 bytes for text blobs."""
        if not requests:
            return {}

        by_repo: dict[str, list[BlobRequest]] = defaultdict(list)
        for req in requests:
            by_repo[req.repo_full_name].append(req)

        out: dict[str, bytes] = {}
        repo_chunks: list[tuple[str, list[BlobRequest]]] = []
        for full_name, files in by_repo.items():
            for i in range(0, len(files), max_files_per_repo):
                repo_chunks.append((full_name, files[i : i + max_files_per_repo]))

        for batch_start in range(0, len(repo_chunks), max_repos_per_query):
            batch = repo_chunks[batch_start : batch_start + max_repos_per_query]
            groups: list[tuple[str, str, str, list[tuple[str, BlobRequest]]]] = []
            for ridx, (full_name, files) in enumerate(batch):
                owner, name = _split_repo(full_name)
                repo_alias = f"repo_{batch_start + ridx}"
                fields = [
                    (f"f_{j}", req)
                    for j, req in enumerate(files)
                    if req.commit_ref
                ]
                if not fields:
                    continue
                groups.append((repo_alias, owner, name, fields))

            if not groups:
                continue
            query = build_batch_query(groups)
            data = self._post(query)

            for repo_alias, _owner, _name, fields in groups:
                repo_data = data.get(repo_alias) or {}
                for field_alias, req in fields:
                    blob = repo_data.get(field_alias)
                    if not isinstance(blob, dict):
                        continue
                    if blob.get("isBinary"):
                        continue
                    if blob.get("isTruncated"):
                        continue
                    text = blob.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    out[req.cache_key] = text.encode("utf-8")

        return out
