"""GraphQL batch query builder tests."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import EmailMessage

import pytest

from llm_dataprep.github_harvest_graphql import BlobRequest, build_batch_query


def _hdrs(**pairs: str) -> EmailMessage:
    msg = EmailMessage()
    for key, val in pairs.items():
        msg[key] = val
    return msg


class _FakeResp:
    def __init__(self, body: bytes, headers: EmailMessage | None = None) -> None:
        self._body = body
        self.headers = headers or EmailMessage()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_build_batch_query_includes_blob_fields() -> None:
    req = BlobRequest(
        repo_full_name="owner/repo",
        path=".pi/agent/sessions/foo.jsonl",
        blob_sha="abc123",
        commit_ref="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        cache_key="owner/repo:.pi/agent/sessions/foo.jsonl",
    )
    q = build_batch_query(
        [
            (
                "repo_0",
                "owner",
                "repo",
                [("f_0", req)],
            )
        ]
    )
    assert "repository(owner: \"owner\", name: \"repo\")" in q
    assert "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef:.pi/agent/sessions/foo.jsonl" in q
    assert "on Blob" in q
    assert "isTruncated" in q


def test_fetch_batch_skips_truncated_blob(monkeypatch) -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    req = BlobRequest(
        repo_full_name="owner/repo",
        path=".pi/agent/sessions/foo.jsonl",
        blob_sha="abc123",
        commit_ref="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        cache_key="k1",
    )
    fetcher = GraphQLBlobFetcher("tok")

    def fake_post(_query: str) -> dict:
        return {
            "repo_0": {
                "f_0": {
                    "isBinary": False,
                    "isTruncated": True,
                    "text": '{"role":"user"}',
                }
            }
        }

    monkeypatch.setattr(fetcher, "_post", fake_post)
    assert fetcher.fetch_batch([req]) == {}


def test_post_returns_partial_data_with_field_errors(monkeypatch) -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    fetcher = GraphQLBlobFetcher("tok")
    payload = json.dumps(
        {
            "data": {"repo_0": {"f_0": {"text": "ok"}}},
            "errors": [{"message": "field error"}],
        }
    ).encode("utf-8")

    class FakeResp:
        headers = {}

        def read(self) -> bytes:
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_graphql.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    data = fetcher._post("query { x }")
    assert data["repo_0"]["f_0"]["text"] == "ok"


def test_fetch_batch_skips_empty_commit_ref() -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    req = BlobRequest(
        repo_full_name="owner/repo",
        path="foo.jsonl",
        blob_sha="abc123",
        commit_ref="",
        cache_key="k-empty",
    )
    fetcher = GraphQLBlobFetcher("tok")
    assert fetcher.fetch_batch([req]) == {}


def test_post_retries_on_http_200_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    calls: list[int] = []
    rate_limited = json.dumps(
        {
            "data": None,
            "errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded."}],
        }
    ).encode()
    ok = json.dumps({"data": {"repo_0": {"f_0": {"text": "ok"}}}}).encode()

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResp:
        calls.append(1)
        if len(calls) == 1:
            return _FakeResp(
                rate_limited,
                _hdrs(
                    **{
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": "9999999999",
                    }
                ),
            )
        return _FakeResp(ok, _hdrs())

    slept: list[float] = []
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_graphql.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("llm_dataprep.github_harvest_graphql.time.sleep", lambda s: slept.append(s))

    fetcher = GraphQLBlobFetcher("tok", max_retries=2)
    data = fetcher._post("query { x }")
    assert data["repo_0"]["f_0"]["text"] == "ok"
    assert len(calls) == 2
    assert len(slept) == 1


def test_post_retries_on_403_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    calls: list[int] = []

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResp:
        calls.append(1)
        if len(calls) == 1:
            err = urllib.error.HTTPError(
                url="https://api.github.com/graphql",
                code=403,
                msg="Forbidden",
                hdrs=_hdrs(**{"Retry-After": "0"}),
                fp=io.BytesIO(b"rate limit"),
            )
            raise err
        return _FakeResp(
            json.dumps({"data": {"repo_0": {"f_0": {"text": "ok"}}}}).encode(),
            _hdrs(),
        )

    slept: list[float] = []
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_graphql.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("llm_dataprep.github_harvest_graphql.time.sleep", lambda s: slept.append(s))

    fetcher = GraphQLBlobFetcher("tok", max_retries=2)
    data = fetcher._post("query { x }")
    assert data["repo_0"]["f_0"]["text"] == "ok"
    assert len(calls) == 2
    assert slept == [1.0]


def test_post_retries_on_429_with_reset_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_dataprep.github_harvest_graphql import GraphQLBlobFetcher

    calls: list[int] = []
    reset_at = "9999999999"

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResp:
        calls.append(1)
        if len(calls) == 1:
            err = urllib.error.HTTPError(
                url="https://api.github.com/graphql",
                code=429,
                msg="Too Many Requests",
                hdrs=_hdrs(**{"X-RateLimit-Reset": reset_at}),
                fp=io.BytesIO(b"rate limit"),
            )
            raise err
        return _FakeResp(
            json.dumps({"data": {"ok": True}}).encode(),
            _hdrs(),
        )

    slept: list[float] = []
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_graphql.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("llm_dataprep.github_harvest_graphql.time.sleep", lambda s: slept.append(s))

    fetcher = GraphQLBlobFetcher("tok", max_retries=2)
    data = fetcher._post("query { x }")
    assert data == {"ok": True}
    assert len(calls) == 2
    assert len(slept) == 1
    assert slept[0] >= 1.0
