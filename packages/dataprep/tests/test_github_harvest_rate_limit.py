"""Rate-limit helpers for GitHub harvest."""

from __future__ import annotations

import time

from llm_dataprep.github_harvest import (
    _backoff_seconds,
    _cache_hit,
    _parse_rate_limit,
    commit_ref_from_html_url,
)


def test_backoff_prefers_retry_after() -> None:
    assert _backoff_seconds(0, retry_after="30", reset_at=None) == 30.0


def test_backoff_exponential_without_headers() -> None:
    assert _backoff_seconds(2, retry_after=None, reset_at=None, base=2.0) == 8.0


def test_parse_rate_limit_headers() -> None:
    snap = _parse_rate_limit(
        {
            "x-ratelimit-remaining": "4999",
            "x-ratelimit-limit": "5000",
            "x-ratelimit-reset": str(int(time.time()) + 60),
            "x-ratelimit-resource": "core",
        }
    )
    assert snap.remaining == 4999
    assert snap.limit == 5000
    assert snap.resource == "core"


def test_cache_hit_same_sha() -> None:
    entry = {"sha": "abc123", "reason": "path"}
    assert _cache_hit(entry, "abc123")
    assert not _cache_hit(entry, "def456")
    assert not _cache_hit(None, "abc123")


def test_commit_ref_from_html_url() -> None:
    url = (
        "https://github.com/SomeRandmGuyy/Hyperstratum/blob/"
        "23630b021ff0f3bc9c9b6a3b0f6aaab71a18a841/"
        ".claude/projects/foo/bar.jsonl"
    )
    assert commit_ref_from_html_url(url) == "23630b021ff0f3bc9c9b6a3b0f6aaab71a18a841"
    assert commit_ref_from_html_url("") == ""


def test_code_search_bucket_does_not_trigger_core_sleep(monkeypatch) -> None:
    from llm_dataprep.github_harvest import GitHubClient

    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("llm_dataprep.github_harvest.time.sleep", fake_sleep)
    client = GitHubClient("tok", low_remaining_threshold=100)
    client._rate_limits["code_search"] = _parse_rate_limit(
        {
            "x-ratelimit-remaining": "8",
            "x-ratelimit-limit": "10",
            "x-ratelimit-reset": str(int(time.time()) + 60),
            "x-ratelimit-resource": "code_search",
        }
    )
    client._proactive_throttle(for_search=True)
    assert slept == []

    client._rate_limits["core"] = _parse_rate_limit(
        {
            "x-ratelimit-remaining": "8",
            "x-ratelimit-limit": "5000",
            "x-ratelimit-reset": str(int(time.time()) + 60),
            "x-ratelimit-resource": "core",
        }
    )
    client._proactive_throttle(for_search=False)
    assert len(slept) == 1
