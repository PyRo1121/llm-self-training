"""GitHub App auth + dual harvest lane tests."""

from __future__ import annotations

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
    API_VERSION,
    GitHubAppConfig,
    InstallationTokenProvider,
    _parse_expires_at,
    app_config_from_env,
    create_installation_access_token,
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
