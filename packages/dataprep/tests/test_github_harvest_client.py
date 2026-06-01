"""Mocked HTTP tests for GitHubClient (_request, code_search, fetch_file_bytes)."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import EmailMessage

import pytest

from llm_dataprep.github_harvest import (
    API_ROOT,
    RAW_CDN_ROOT,
    GitHubClient,
    GitHubFetchError,
)


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


def test_request_parses_json_and_applies_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"ok": True}).encode("utf-8")
    headers = _hdrs(
        **{
            "X-RateLimit-Remaining": "9",
            "X-RateLimit-Limit": "10",
            "X-RateLimit-Reset": "9999999999",
            "X-RateLimit-Resource": "code_search",
        }
    )

    monkeypatch.setattr(
        "llm_dataprep.github_harvest.urllib.request.urlopen",
        lambda *a, **k: _FakeResp(payload, headers),
    )
    client = GitHubClient("tok")
    data, resp_hdr = client._request(f"{API_ROOT}/search/code?q=test")
    assert data == {"ok": True}
    assert resp_hdr["x-ratelimit-remaining"] == "9"
    snap = client.rate_limit_status()
    assert snap.remaining == 9
    assert snap.resource == "code_search"


def test_request_download_404_raises_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_404(req: object, timeout: float = 120) -> None:
        err = urllib.error.HTTPError(
            url=f"{API_ROOT}/repos/o/r/git/blobs/deadbeef",
            code=404,
            msg="Not Found",
            hdrs=_hdrs(),
            fp=io.BytesIO(b'{"message":"Not Found"}'),
        )
        raise err

    monkeypatch.setattr("llm_dataprep.github_harvest.urllib.request.urlopen", raise_404)
    client = GitHubClient("tok", max_retries=1)
    with pytest.raises(GitHubFetchError, match="HTTP 404") as exc_info:
        client._request(
            f"{API_ROOT}/repos/o/r/git/blobs/deadbeef",
            accept="application/vnd.github.raw",
            raw_bytes=True,
        )
    assert exc_info.value.gone is True


def test_request_retries_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_urlopen(req: object, timeout: float = 120) -> _FakeResp:
        calls.append(1)
        if len(calls) == 1:
            err = urllib.error.HTTPError(
                url=str(req),
                code=403,
                msg="Forbidden",
                hdrs=_hdrs(**{"Retry-After": "0"}),
                fp=io.BytesIO(b"rate limit"),
            )
            raise err
        return _FakeResp(b'{"items":[]}', _hdrs())

    slept: list[float] = []
    monkeypatch.setattr("llm_dataprep.github_harvest.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("llm_dataprep.github_harvest.time.sleep", lambda s: slept.append(s))

    client = GitHubClient("tok", max_retries=2)
    data, _hdr = client._request(f"{API_ROOT}/search/code?q=x")
    assert data == {"items": []}
    assert len(calls) == 2
    assert slept == [1.0]


def test_code_search_uses_text_match_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[urllib.request.Request] = []
    body = json.dumps({"items": [{"path": "a.jsonl"}], "total_count": 1}).encode()

    def fake_urlopen(req: urllib.request.Request, timeout: float = 120) -> _FakeResp:
        captured.append(req)
        return _FakeResp(
            body,
            _hdrs(
                **{
                    "X-RateLimit-Remaining": "8",
                    "X-RateLimit-Limit": "10",
                    "X-RateLimit-Resource": "code_search",
                }
            ),
        )

    monkeypatch.setattr("llm_dataprep.github_harvest.urllib.request.urlopen", fake_urlopen)
    client = GitHubClient("tok", lane="pat")
    result = client.code_search("extension:jsonl foo", page=2, per_page=50)
    assert result["total_count"] == 1
    assert len(captured) == 1
    req = captured[0]
    assert req.get_full_url().startswith(f"{API_ROOT}/search/code?")
    assert "q=extension%3Ajsonl+foo" in req.get_full_url()
    assert "page=2" in req.get_full_url()
    assert "per_page=50" in req.get_full_url()
    assert req.get_header("Accept") == "application/vnd.github.text-match+json"
    assert req.get_header("Authorization") == "Bearer tok"


def test_fetch_file_bytes_via_git_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b'{"role":"user","text":"hi"}\n'

    monkeypatch.setattr(
        "llm_dataprep.github_harvest.urllib.request.urlopen",
        lambda *a, **k: _FakeResp(raw, _hdrs()),
    )
    client = GitHubClient("tok", use_raw_cdn=False)
    out = client.fetch_file_bytes(
        "owner/repo",
        ".pi/sessions/foo.jsonl",
        blob_sha="abc123sha",
        min_interval=0.0,
    )
    assert out == raw


def test_fetch_file_bytes_cdn_skips_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[urllib.request.Request] = []
    ref = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    cdn_url = f"{RAW_CDN_ROOT}/owner/repo/{ref}/path/file.jsonl"

    def fake_urlopen(req: urllib.request.Request, timeout: float = 120) -> _FakeResp:
        captured.append(req)
        if "/git/blobs/" in req.get_full_url():
            err = urllib.error.HTTPError(
                url=req.get_full_url(),
                code=404,
                msg="Not Found",
                hdrs=_hdrs(),
                fp=io.BytesIO(b"gone"),
            )
            raise err
        return _FakeResp(b"from-cdn", _hdrs())

    monkeypatch.setattr("llm_dataprep.github_harvest.urllib.request.urlopen", fake_urlopen)
    client = GitHubClient("tok")
    out = client.fetch_file_bytes(
        "owner/repo",
        "path/file.jsonl",
        blob_sha="missing",
        commit_ref=ref,
        min_interval=0.0,
    )
    assert out == b"from-cdn"
    assert len(captured) == 2
    assert captured[1].get_full_url() == cdn_url
    assert not captured[1].has_header("Authorization")


def test_fetch_file_bytes_contents_raw_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float = 120) -> _FakeResp:
        url = req.get_full_url()
        if "/git/blobs/" in url or url.startswith(RAW_CDN_ROOT):
            err = urllib.error.HTTPError(
                url=url,
                code=404,
                msg="Not Found",
                hdrs=_hdrs(),
                fp=io.BytesIO(b"gone"),
            )
            raise err
        return _FakeResp(b"from-contents-api", _hdrs())

    monkeypatch.setattr("llm_dataprep.github_harvest.urllib.request.urlopen", fake_urlopen)
    client = GitHubClient("tok")
    out = client.fetch_file_bytes(
        "owner/repo",
        "path/file.jsonl",
        blob_sha="gone",
        commit_ref="abc123",
        min_interval=0.0,
    )
    assert out == b"from-contents-api"
