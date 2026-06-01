"""Harvest state cache — Redis primary, JSON file fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

CACHE_PREFIX = "llm:github_harvest:"


def redis_url_from_env() -> str | None:
    url = os.environ.get("REDIS_URL", "").strip()
    if url:
        return url
    password = os.environ.get("REDIS_PASSWORD", "").strip()
    host = os.environ.get("REDIS_HOST", "127.0.0.1").strip()
    port = os.environ.get("REDIS_PORT", "6380").strip()
    db = os.environ.get("REDIS_DB", "0").strip()
    if not password:
        return None
    user = os.environ.get("REDIS_USER", "").strip()
    auth = ""
    if user:
        auth = f"{quote(user, safe='')}:{quote(password, safe='')}"
    else:
        auth = f":{quote(password, safe='')}"
    return f"redis://{auth}@{host}:{port}/{db}"


class HarvestCache:
    """Dual-layer cache: Redis (fast) + state.json (durable fallback)."""

    def __init__(self, state_path: Path, *, redis_url: str | None = None) -> None:
        self._state_path = state_path
        self._redis = None
        self._redis_ok = False
        url = redis_url or redis_url_from_env()
        if url:
            try:
                import redis

                self._redis = redis.from_url(url, decode_responses=True, socket_timeout=5)
                self._redis.ping()
                self._redis_ok = True
            except Exception as exc:
                print(f"github-harvest: Redis unavailable ({exc}) — using JSON state only", flush=True)

        self._state = self._load_json_state()

    @property
    def backend(self) -> str:
        return "redis+json" if self._redis_ok else "json"

    def _load_json_state(self) -> dict[str, Any]:
        if not self._state_path.is_file():
            return {"seen": {}, "rejected": {}, "queries": {}}
        try:
            doc = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"seen": {}, "rejected": {}, "queries": {}}
        doc.setdefault("seen", {})
        doc.setdefault("rejected", {})
        doc.setdefault("queries", {})
        return doc

    def _rkey(self, bucket: str, key: str) -> str:
        return f"{CACHE_PREFIX}{bucket}:{key}"

    def get_entry(self, bucket: str, key: str) -> dict[str, Any] | None:
        if self._redis_ok and self._redis is not None:
            raw = self._redis.get(self._rkey(bucket, key))
            if raw:
                try:
                    val = json.loads(raw)
                    if isinstance(val, dict):
                        return val
                except json.JSONDecodeError:
                    pass
        bucket_map = self._state.get(bucket) or {}
        val = bucket_map.get(key)
        return val if isinstance(val, dict) else None

    def set_entry(self, bucket: str, key: str, value: dict[str, Any]) -> None:
        self._state.setdefault(bucket, {})[key] = value
        if self._redis_ok and self._redis is not None:
            self._redis.set(self._rkey(bucket, key), json.dumps(value, separators=(",", ":")))

    def get_seen(self, key: str) -> dict[str, Any] | None:
        return self.get_entry("seen", key)

    def set_seen(self, key: str, value: dict[str, Any]) -> None:
        self.set_entry("seen", key, value)

    def get_rejected(self, key: str) -> dict[str, Any] | None:
        return self.get_entry("rejected", key)

    def set_rejected(self, key: str, value: dict[str, Any]) -> None:
        self.set_entry("rejected", key, value)

    def get_query_state(self, query_id: str) -> dict[str, Any] | None:
        if self._redis_ok and self._redis is not None:
            raw = self._redis.get(self._rkey("queries", query_id))
            if raw:
                try:
                    val = json.loads(raw)
                    if isinstance(val, dict):
                        return val
                except json.JSONDecodeError:
                    pass
        bucket = self._state.get("queries") or {}
        val = bucket.get(query_id)
        return val if isinstance(val, dict) else None

    def set_query_state(self, query_id: str, value: dict[str, Any]) -> None:
        self._state.setdefault("queries", {})[query_id] = value
        if self._redis_ok and self._redis is not None:
            self._redis.set(
                self._rkey("queries", query_id),
                json.dumps(value, separators=(",", ":")),
            )

    def flush(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(self._state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def clear_all(self) -> None:
        self._state = {"seen": {}, "rejected": {}, "queries": {}}
        if self._redis_ok and self._redis is not None:
            batch: list[str] = []
            for key in self._redis.scan_iter(match=f"{CACHE_PREFIX}*"):
                batch.append(key)
                if len(batch) >= 500:
                    self._redis.delete(*batch)
                    batch.clear()
            if batch:
                self._redis.delete(*batch)
        self.flush()

    @property
    def seen(self) -> dict[str, Any]:
        return self._state.setdefault("seen", {})

    @property
    def rejected(self) -> dict[str, Any]:
        return self._state.setdefault("rejected", {})
