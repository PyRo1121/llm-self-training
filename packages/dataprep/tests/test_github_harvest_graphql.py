"""GraphQL batch query builder tests."""

from __future__ import annotations

import json

from llm_dataprep.github_harvest_graphql import BlobRequest, build_batch_query


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
