"""HarvestCache — Redis/JSON dual-layer behavior."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.github_harvest_cache import CACHE_PREFIX, HarvestCache


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def scan_iter(self, *, match: str):
        prefix = match.rstrip("*")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)
            self.deleted.append(key)


def test_get_query_state_reads_redis_before_json(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"seen": {}, "rejected": {}, "queries": {}}', encoding="utf-8")
    cache = HarvestCache(state_path)
    fake = _FakeRedis()
    cache._redis = fake
    cache._redis_ok = True

    cache.set_query_state("q1", {"last_page": 3, "pages": 5})
    cache._state["queries"] = {}

    got = cache.get_query_state("q1")
    assert got is not None
    assert got["last_page"] == 3


def test_get_entry_requires_dict_from_redis(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"seen": {"k1": {"repo": "acme/foo"}}, "rejected": {}, "queries": {}}',
        encoding="utf-8",
    )
    cache = HarvestCache(state_path)
    fake = _FakeRedis()
    fake.store[f"{CACHE_PREFIX}seen:k1"] = json.dumps(["not", "a", "dict"])
    cache._redis = fake
    cache._redis_ok = True

    got = cache.get_entry("seen", "k1")
    assert got == {"repo": "acme/foo"}


def test_get_entry_returns_dict_from_redis(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    cache = HarvestCache(state_path)
    fake = _FakeRedis()
    cache._redis = fake
    cache._redis_ok = True

    cache.set_seen("k1", {"repo": "acme/bar", "sha": "abc"})
    cache._state["seen"] = {}

    got = cache.get_seen("k1")
    assert got == {"repo": "acme/bar", "sha": "abc"}


def test_clear_all_deletes_in_batches(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    cache = HarvestCache(state_path)
    fake = _FakeRedis()
    for i in range(1200):
        fake.store[f"{CACHE_PREFIX}seen:k{i}"] = "{}"
    cache._redis = fake
    cache._redis_ok = True

    cache.clear_all()
    assert fake.store == {}
    assert len(fake.deleted) == 1200


def test_clear_all_persists_empty_state_json(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"seen": {"k1": {"repo": "acme/foo"}}, "rejected": {}, "queries": {"q1": {"page": 1}}}',
        encoding="utf-8",
    )
    cache = HarvestCache(state_path)
    cache._state["seen"]["k1"] = {"repo": "acme/foo"}
    cache._state["queries"]["q1"] = {"page": 1}

    cache.clear_all()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted == {"seen": {}, "rejected": {}, "queries": {}}
    assert cache.seen == {}
    assert cache.rejected == {}
