"""GitHub harvest registry + path regex filters."""

from __future__ import annotations

import json

from llm_dataprep.github_harvest import (
    HarvestConfig,
    looks_like_chat_blob,
    should_accept_path,
)
from llm_dataprep.github_harvest_registry import (
    HARVEST_QUERY_REGISTRY,
    registry_queries,
)


def test_registry_covers_major_harnesses() -> None:
    hints = {q["harness_hint"] for q in HARVEST_QUERY_REGISTRY}
    for expected in (
        "cursor",
        "codex",
        "claude_code",
        "pi",
        "opencode",
        "gemini_cli",
        "copilot",
        "kimi",
        "factory",
        "openclaw",
    ):
        assert expected in hints


def test_registry_has_33_queries() -> None:
    assert len(HARVEST_QUERY_REGISTRY) == 33


def test_registry_queries_tier_a_sorted_first() -> None:
    qs = registry_queries()
    tier_ids = {q["id"] for q in qs if q.get("priority", 50) <= 10}
    assert "cursor_agent_transcripts" in tier_ids
    assert qs[0]["id"] == "claude_code_projects" or qs[0].get("priority") == 10
    cursor = next(q for q in qs if q["id"] == "cursor_agent_transcripts")
    assert cursor.get("max_pages") == 10


def test_copilot_chronicle_requires_dot_copilot_path() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "copilot_chronicle_events")
    assert should_accept_path(".copilot/session-state/abc/events.jsonl", qspec, cfg)
    assert not should_accept_path("session-state/abc/events.jsonl", qspec, cfg)


def test_harbor_rollout_rejected_by_codex_regex() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "codex_rollout_sessions")
    harbor = (
        "jobs/daytona/agent/sessions/2026/03/02/"
        "rollout-2026-03-02T05-42-18-019cad11.jsonl"
    )
    assert not should_accept_path(harbor, qspec, cfg)
    ok = ".codex/sessions/2026/03/02/rollout-2026-03-02T05-42-18.jsonl"
    assert should_accept_path(ok, qspec, cfg)
    assert should_accept_path(f"./{ok}", qspec, cfg)


def test_pi_path_regex() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "pi_agent_sessions")
    assert should_accept_path(
        ".pi/agent/sessions/--home-user-proj--/20260228_143022_abc123.jsonl",
        qspec,
        cfg,
    )
    assert not should_accept_path("docs/pi/agent/sessions/fake.jsonl", qspec, cfg)


def test_cursor_path_accepts_nested_or_flat() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "cursor_agent_transcripts")
    assert should_accept_path(
        ".cursor/projects/p/agent-transcripts/uuid/uuid.jsonl",
        qspec,
        cfg,
    )
    assert should_accept_path(
        ".cursor/projects/p/agent-transcripts/session.jsonl",
        qspec,
        cfg,
    )
    assert not should_accept_path(
        ".cursor/projects/p/agent-transcripts/subagents/u.jsonl",
        qspec,
        cfg,
    )


def test_opencode_path_regex() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "opencode_storage_message")
    assert should_accept_path(
        ".local/share/opencode/storage/message/sess123/user.json",
        qspec,
        cfg,
    )


def test_cline_rejects_config_json() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "cline_tasks")
    assert not should_accept_path(".cline/data/globalState.json", qspec, cfg)
    assert not should_accept_path(".cline/hooks.json", qspec, cfg)
    assert should_accept_path(".cline/data/tasks/123/api_conversation_history.json", qspec, cfg)
    assert should_accept_path(
        "Code/User/globalStorage/saoudrizwan.claude-dev/tasks/abc123/api_conversation_history.json",
        qspec,
        cfg,
    )


def test_roo_vscode_tasks_path() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "roo_code_tasks")
    assert should_accept_path(
        "Code/User/globalStorage/rooveterinaryinc.roo-cline/tasks/abc123/api_conversation_history.json",
        qspec,
        cfg,
    )


def test_goose_sessions_jsonl_path() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "goose_sessions_jsonl")
    assert should_accept_path(".local/share/goose/sessions/20250310_2.jsonl", qspec, cfg)
    assert not should_accept_path(".goose/proj/conversation.json", qspec, cfg)


def test_kiro_cli_sessions_extension_queries() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    jsonl = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "kiro_cli_sessions")
    json_q = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "kiro_cli_sessions_json")
    assert "extension:jsonl" in jsonl["q"]
    assert "extension:json" in json_q["q"]
    assert "extension:jsonl" not in json_q["q"]
    assert should_accept_path(".kiro/sessions/cli/sess-abc.jsonl", jsonl, cfg)
    assert not should_accept_path(".kiro/sessions/cli/sess-abc.json", jsonl, cfg)
    assert should_accept_path(".kiro/sessions/cli/sess-abc.json", json_q, cfg)
    assert not should_accept_path(".kiro/sessions/cli/sess-abc.jsonl", json_q, cfg)


def test_claude_session_filename_regex() -> None:
    cfg = HarvestConfig(exclude_path_regex=())
    qspec = next(q for q in HARVEST_QUERY_REGISTRY if q["id"] == "claude_session_filename")
    assert should_accept_path("logs/claude-session.jsonl", qspec, cfg)
    assert not should_accept_path("logs/claude-session.json", qspec, cfg)


def test_looks_like_claude_single_line_agent() -> None:
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Ready to help."}],
            },
        }
    )
    assert looks_like_chat_blob(line, "claude_code", min_hits=2)


def test_looks_like_pi_message_jsonl() -> None:
    lines = [
        json.dumps({"type": "session", "id": "x", "cwd": "/tmp"}),
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "fix the bug please"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll inspect the repo."}],
                },
            }
        ),
    ]
    assert looks_like_chat_blob("\n".join(lines), "pi")


def test_load_config_uses_registry(tmp_path, monkeypatch) -> None:
    from llm_dataprep.github_harvest import load_harvest_config

    cfg_file = tmp_path / "github-harvest.yaml"
    cfg_file.write_text(
        """
github_harvest:
  queries: registry
  disabled_queries: [aider_chat_history]
""",
        encoding="utf-8",
    )
    cfg = load_harvest_config(cfg_file)
    assert len(cfg.queries) == len(registry_queries(disabled=("aider_chat_history",)))
    assert all(q["id"] != "aider_chat_history" for q in cfg.queries)
