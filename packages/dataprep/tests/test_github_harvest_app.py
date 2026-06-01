"""GitHub App auth + dual harvest lane tests."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_dataprep.github_harvest import (
    HarvestConfig,
    HarvestLane,
    build_harvest_lanes,
    iter_code_search_hits,
    pick_download_lane,
    pick_download_lane_for_hit,
    pick_search_lane,
)
from llm_dataprep.github_harvest_app import (
    API_ROOT,
    API_VERSION,
    GitHubAppConfig,
    InstallationTokenProvider,
    _app_api_request,
    _load_private_key_pem,
    _parse_expires_at,
    app_config_from_env,
    create_installation_access_token,
    list_installations,
    make_app_jwt,
    resolve_user_installation_id,
    token_provider_from_env,
)

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402


@pytest.fixture(scope="module")
def rsa_private_key_pem() -> bytes:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def test_pick_search_lane_alternates() -> None:
    pat = HarvestLane("pat", MagicMock(), None, search=True)
    app = HarvestLane("app", MagicMock(), None, search=True)
    assert pick_search_lane([pat, app], 0).name == "pat"
    assert pick_search_lane([pat, app], 1).name == "app"
    assert pick_search_lane([pat, app], 2).name == "pat"


def test_iter_code_search_alternates_lanes_per_page() -> None:
    pat_client = MagicMock()
    app_client = MagicMock()
    pat_client._lane = "pat"
    app_client._lane = "app"
    item = {
        "repository": {"full_name": "o/r"},
        "path": "a.jsonl",
        "sha": "s1",
        "html_url": "https://github.com/o/r/blob/main/a.jsonl",
    }
    pat_client.code_search_rate_limited.return_value = {"items": [item] * 100}
    app_client.code_search_rate_limited.return_value = {"items": [{**item, "path": "b.jsonl", "sha": "s2"}]}
    pat = HarvestLane("pat", pat_client, None, search=True)
    app = HarvestLane("app", app_client, None, search=True)
    cfg = HarvestConfig(default_max_pages=2)
    qspec = {"id": "t", "q": "filename:tasks.jsonl"}
    counter = [0]
    hits = list(
        iter_code_search_hits(
            None,
            qspec,
            cfg,
            lanes=[pat, app],
            search_request_counter=counter,
            max_pages=2,
        )
    )
    assert counter[0] == 2
    assert pat_client.code_search_rate_limited.call_count == 1
    assert app_client.code_search_rate_limited.call_count == 1
    assert hits[0].auth_lane == "pat"
    assert hits[100].auth_lane == "app"


def test_pick_download_lane_for_hit_prefers_app_download_only() -> None:
    pat = HarvestLane("pat", MagicMock(), None, search=True)
    app = HarvestLane("app", MagicMock(), None, search=False)
    assert pick_download_lane_for_hit([pat, app], "pat").name == "app"


def test_pick_download_lane_app_download_only() -> None:
    pat = HarvestLane("pat", MagicMock(), None, search=True)
    app = HarvestLane("app", MagicMock(), None, search=False)
    assert pick_download_lane([pat, app], pat).name == "app"


def test_app_config_from_env_invalid_installation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "not-an-int")
    monkeypatch.setenv(
        "GITHUB_APP_PRIVATE_KEY",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----",
    )
    with pytest.warns(UserWarning, match="GITHUB_APP_INSTALLATION_ID"):
        assert app_config_from_env() is None


def test_api_version_is_current() -> None:
    assert API_VERSION == "2026-03-10"


def test_app_config_from_env_missing_key_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(tmp_path / "missing.pem"))
    with pytest.warns(UserWarning, match="GITHUB_APP_PRIVATE_KEY_PATH"):
        assert app_config_from_env() is None


def test_build_harvest_lanes_pat_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_APP_CLIENT_ID", raising=False)
    cfg = HarvestConfig()
    lanes = build_harvest_lanes(cfg, "ghp_test")
    assert len(lanes) == 1
    assert lanes[0].name == "pat"


def test_build_harvest_lanes_dual(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "99")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----")

    cfg = HarvestConfig()
    with patch(
        "llm_dataprep.github_harvest.token_provider_from_env",
        return_value=lambda: "ghs_installation_token",
    ):
        lanes = build_harvest_lanes(cfg, "ghp_pat")
    assert len(lanes) == 2
    assert {lane.name for lane in lanes} == {"pat", "app"}


def test_parse_expires_at_iso() -> None:
    ts = _parse_expires_at("2030-01-01T00:00:00Z")
    assert ts >= 1_893_456_000


def test_create_installation_access_token_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.make_app_jwt",
        lambda **_: "jwt-test",
    )
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        lambda *a, **k: {"token": "ghs_abc", "expires_at": "2030-01-01T00:00:00Z"},
    )
    token, exp = create_installation_access_token(
        client_id="Iv1.test",
        private_key_pem=b"key",
        installation_id=1,
    )
    assert token == "ghs_abc"
    assert exp > 0


def test_installation_token_provider_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_create(**_: object) -> tuple[str, float]:
        calls["n"] += 1
        return f"tok{calls['n']}", 9_999_999_999.0

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.create_installation_access_token",
        fake_create,
    )
    cfg = GitHubAppConfig(
        client_id="Iv1.test",
        installation_id=1,
        private_key_pem=b"k",
    )
    provider = InstallationTokenProvider(cfg)
    assert provider() == "tok1"
    assert provider() == "tok1"
    provider._expires_at = 0
    assert provider() == "tok2"


def test_make_app_jwt_rs256(rsa_private_key_pem: bytes) -> None:
    import jwt

    token = make_app_jwt(client_id="Iv1.testclient", private_key_pem=rsa_private_key_pem)
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["iss"] == "Iv1.testclient"
    assert payload["exp"] - payload["iat"] <= 600


def test_make_app_jwt_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "jwt":
            raise ImportError("no jwt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="PyJWT"):
        make_app_jwt(client_id="Iv1.test", private_key_pem=b"key")


def test_load_private_key_pem_inline_escapes_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inline = "-----BEGIN RSA PRIVATE KEY-----\\nMIIB\\n-----END RSA PRIVATE KEY-----"
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", inline)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
    pem = _load_private_key_pem()
    assert pem == b"-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----"


def test_app_config_from_env_inline_pem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "4242")
    monkeypatch.setenv(
        "GITHUB_APP_PRIVATE_KEY",
        "-----BEGIN RSA PRIVATE KEY-----\\nMIIB\\n-----END RSA PRIVATE KEY-----",
    )
    monkeypatch.setenv("GITHUB_APP_DOWNLOAD_ONLY", "yes")
    cfg = app_config_from_env()
    assert cfg is not None
    assert cfg.client_id == "Iv1.test"
    assert cfg.installation_id == 4242
    assert b"\\n" not in cfg.private_key_pem
    assert cfg.download_only is True


def test_app_config_from_env_key_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rsa_private_key_pem: bytes
) -> None:
    key_path = tmp_path / "app.pem"
    key_path.write_bytes(rsa_private_key_pem)
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "7")
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(key_path))
    cfg = app_config_from_env()
    assert cfg is not None
    assert cfg.private_key_pem == rsa_private_key_pem


def test_app_api_request_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"token": "ghs_ok"}).encode("utf-8")

    class FakeResp:
        def read(self) -> bytes:
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    doc = _app_api_request(
        "POST",
        f"{API_ROOT}/app/installations/1/access_tokens",
        jwt_token="jwt-test",
        body={"permissions": {"contents": "read"}},
    )
    assert doc["token"] == "ghs_ok"


def test_app_api_request_empty_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    assert _app_api_request("GET", f"{API_ROOT}/app/installations", jwt_token="jwt") == {}


def test_app_api_request_404_access_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_404(*args: object, **kwargs: object) -> None:
        err = urllib.error.HTTPError(
            url=f"{API_ROOT}/app/installations/999/access_tokens",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"Not Found"}'),
        )
        raise err

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        raise_404,
    )
    with pytest.raises(RuntimeError, match="wrong GITHUB_APP_INSTALLATION_ID"):
        _app_api_request(
            "POST",
            f"{API_ROOT}/app/installations/999/access_tokens",
            jwt_token="jwt-test",
        )


def test_app_api_request_non404_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_500(*args: object, **kwargs: object) -> None:
        err = urllib.error.HTTPError(
            url=f"{API_ROOT}/app/installations",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=io.BytesIO(b"internal error"),
        )
        raise err

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        raise_500,
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        _app_api_request("GET", f"{API_ROOT}/app/installations", jwt_token="jwt")


def test_resolve_user_installation_id_success(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        lambda method, url, **kw: {"id": 555},
    )
    iid = resolve_user_installation_id(
        client_id="Iv1.test",
        private_key_pem=rsa_private_key_pem,
        username="octocat",
    )
    assert iid == 555


def test_resolve_user_installation_id_404(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    def raise_404(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("GitHub App API HTTP 404 for https://api.github.com/users/x/installation: nf")

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        raise_404,
    )
    with pytest.raises(RuntimeError, match="not installed on user"):
        resolve_user_installation_id(
            client_id="Iv1.test",
            private_key_pem=rsa_private_key_pem,
            username="nobody",
        )


def test_resolve_user_installation_id_missing_id(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        lambda *a, **k: {"account": {"login": "octocat"}},
    )
    with pytest.raises(RuntimeError, match="No app installation found"):
        resolve_user_installation_id(
            client_id="Iv1.test",
            private_key_pem=rsa_private_key_pem,
            username="octocat",
        )


def test_list_installations_success(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    rows = [{"id": 1, "account": {"login": "a"}}, {"id": 2, "account": {"login": "b"}}]
    payload = json.dumps(rows).encode("utf-8")

    class FakeResp:
        def read(self) -> bytes:
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    out = list_installations(client_id="Iv1.test", private_key_pem=rsa_private_key_pem)
    assert out == rows


def test_list_installations_non_list_response(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return json.dumps({"message": "unexpected"}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    assert list_installations(client_id="Iv1.test", private_key_pem=rsa_private_key_pem) == []


def test_list_installations_http_error(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes
) -> None:
    def raise_403(*args: object, **kwargs: object) -> None:
        err = urllib.error.HTTPError(
            url=f"{API_ROOT}/app/installations",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b"forbidden"),
        )
        raise err

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.urllib.request.urlopen",
        raise_403,
    )
    with pytest.raises(RuntimeError, match="HTTP 403"):
        list_installations(client_id="Iv1.test", private_key_pem=rsa_private_key_pem)


def test_create_installation_access_token_with_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_request(method: str, url: str, **kw: object) -> dict:
        captured["body"] = kw.get("body")
        return {"token": "ghs_perm", "expires_at": "2030-06-01T00:00:00Z"}

    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.make_app_jwt",
        lambda **_: "jwt-test",
    )
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        fake_request,
    )
    token, _ = create_installation_access_token(
        client_id="Iv1.test",
        private_key_pem=b"key",
        installation_id=3,
        permissions={"contents": "read"},
    )
    assert token == "ghs_perm"
    assert captured["body"] == {"permissions": {"contents": "read"}}


def test_create_installation_access_token_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app.make_app_jwt",
        lambda **_: "jwt-test",
    )
    monkeypatch.setattr(
        "llm_dataprep.github_harvest_app._app_api_request",
        lambda *a, **k: {"expires_at": "2030-01-01T00:00:00Z"},
    )
    with pytest.raises(RuntimeError, match="missing token"):
        create_installation_access_token(
            client_id="Iv1.test",
            private_key_pem=b"key",
            installation_id=1,
        )


def test_parse_expires_at_invalid_fallback() -> None:
    before = _parse_expires_at("not-a-date")
    assert before > 0


def test_token_provider_from_env(
    monkeypatch: pytest.MonkeyPatch, rsa_private_key_pem: bytes, tmp_path: Path
) -> None:
    key_path = tmp_path / "app.pem"
    key_path.write_bytes(rsa_private_key_pem)
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "Iv1.test")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "1")
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(key_path))
    provider = token_provider_from_env()
    assert provider is not None
    assert callable(provider)
