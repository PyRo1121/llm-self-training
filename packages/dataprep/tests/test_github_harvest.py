"""GitHub public session harvest — parsers, filters, config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_dataprep.curate_raw import _public_meta_defaults
from llm_dataprep.github_harvest import (
    CodeHit,
    HarvestConfig,
    detect_harness,
    load_harvest_config,
    looks_like_chat_blob,
    parse_blob_text,
    run_harvest,
    should_accept_path,
    should_skip_path,
    should_skip_repo,
    _flush_pending_downloads_batch,
)
from llm_dataprep.github_harvest_cache import HarvestCache


def _hit(path: str, *, qid: str = "test") -> CodeHit:
    return CodeHit(
        repo_full_name="someone/agent-logs",
        path=path,
        sha="deadbeef",
        html_url="https://github.com/someone/agent-logs/blob/main/" + path,
        query_id=qid,
    )


def test_detect_harness_paths() -> None:
    assert detect_harness(".cursor/projects/p/agent-transcripts/u/u.jsonl") == "cursor"
    assert detect_harness(".copilot/session-state/abc/events.jsonl") == "copilot"
    assert detect_harness("session-state/events.jsonl") == "generic"
    assert detect_harness(
        ".config/Code/User/workspaceStorage/ws/chatSessions/abc.jsonl"
    ) == "copilot_vscode"
    assert detect_harness("jobs/foo/rollout-abc.jsonl") == "generic"
    assert detect_harness(".codex/sessions/2026/03/01/rollout-abc.jsonl") == "codex"
    assert detect_harness(".kimi/sessions/uuid/context.jsonl") == "kimi"
    assert detect_harness(".openclaw/agents/main/sessions/sess.jsonl") == "openclaw"
    assert detect_harness(".clawdbot/sessions/sess.jsonl") == "openclaw"
    assert detect_harness(".factory/projects/foo/bar.jsonl") == "factory"
    assert detect_harness(".gemini/tmp/abc/chats/session-2026.jsonl") == "gemini_cli"
    assert (
        detect_harness(".config/tokscale/antigravity-cache/sessions/s1.jsonl")
        == "antigravity"
    )
    assert detect_harness(".config/tokscale/trae-cache/sessions/s1.json") == "trae"
    assert detect_harness(".qwen/projects/p/chats/session-1.jsonl") == "qwen_cli"
    assert detect_harness("misc/transcripts/foo.jsonl", "cursor") == "cursor"


def test_detect_harness_mux_aider_kiro_continue() -> None:
    assert detect_harness(".mux/sessions/proj/sess.jsonl") == "mux"
    assert detect_harness("project/.aider.chat.history.md") == "aider"
    assert detect_harness(".kiro/sessions/uuid/thread.jsonl") == "kiro"
    assert detect_harness(".continue/sessions/sess-abc.json") == "continue"
    assert detect_harness(".continue/projects/myproj/session.json") == "continue"


def test_detect_harness_amp() -> None:
    assert detect_harness(".local/share/amp/threads/thread-abc.json") == "amp"
    assert detect_harness("home/user/.local/share/amp/threads/thread-abc.json") == "amp"
    assert detect_harness("backup/amp/threads/thread-abc.json") == "amp"


def test_detect_harness_openhands_tightened() -> None:
    assert (
        detect_harness(".openhands-state/sessions/sess-abc/events/42.json")
        == "openhands"
    )
    assert detect_harness("conversations/sess-xyz/events/7.json") == "openhands"
    assert detect_harness(".openhands-state/config.json") == "generic"
    assert detect_harness(".openhands-state/sessions/sess/events.jsonl") == "generic"
    assert (
        detect_harness("conversations/abc/events-handler/message.json") == "generic"
    )
    assert (
        detect_harness("data/conversations/user1/chat/events_processor/out.json")
        == "generic"
    )


def test_should_accept_path_rejects_harbor_jobs() -> None:
    cfg = HarvestConfig(
        exclude_path_substrings=("/jobs/", "harbor/"),
    )
    qspec = {
        "require_path_substrings": (".codex/sessions",),
    }
    harbor = (
        "jobs/daytona/agent/sessions/2026/03/02/"
        "rollout-2026-03-02T05-42-18-019cad11.jsonl"
    )
    assert not should_accept_path(harbor, qspec, cfg)
    ok = ".codex/sessions/2026/03/02/rollout-2026-03-02T05-42-18.jsonl"
    assert should_accept_path(ok, qspec, cfg)


def test_looks_like_chat_blob_cursor() -> None:
    lines = [
        json.dumps(
            {
                "role": "user",
                "message": {"content": [{"type": "text", "text": "hello world"}]},
            }
        ),
        json.dumps(
            {
                "role": "assistant",
                "message": {"content": [{"type": "text", "text": "hi there friend"}]},
            }
        ),
    ]
    assert looks_like_chat_blob("\n".join(lines), "cursor")
    assert not looks_like_chat_blob('{"event":"build"}\n{"status":"ok"}', "cursor")


def test_looks_like_chat_blob_cursor_flat_string_content() -> None:
    lines = [
        json.dumps({"type": "human", "message": {"content": "Fix the auth module please."}}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": "I'll inspect the auth tests and patch the failure."},
            }
        ),
    ]
    assert looks_like_chat_blob("\n".join(lines), "cursor")


def test_looks_like_chat_blob_claude_human_line() -> None:
    line = json.dumps(
        {
            "type": "human",
            "message": "Add retry logic to the GitHub harvest client.",
        }
    )
    assert looks_like_chat_blob(line, "claude_code", min_hits=2)


def test_parse_cursor_flat_path_blob() -> None:
    hit = _hit(".cursor/projects/p/agent-transcripts/session.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "human", "message": {"content": "Explain this pytest failure."}}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": "The assertion fails because the mock was never called."},
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="cursor", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[0]["harness"] == "cursor"
    assert "pytest failure" in recs[0]["text"]


def test_parse_claude_human_line_blob() -> None:
    hit = _hit(".claude/projects/abc/session.jsonl")
    blob = "\n".join(
        [
            json.dumps(
                {
                    "type": "human",
                    "message": "Wire claude_code harvest to accept human turns.",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": "Updated _role_and_text to map human to user.",
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="claude_code", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[0]["harness"] == "claude_code"
    assert "human turns" in recs[0]["text"]


def test_should_skip_path_and_repo() -> None:
    cfg = HarvestConfig(
        exclude_path_substrings=("node_modules/", "/dist/"),
        exclude_repo_prefixes=("PyRo1121/",),
    )
    assert should_skip_path("foo/node_modules/bar.jsonl", cfg)
    assert should_skip_path("src/dist/out.jsonl", cfg)
    assert not should_skip_path("agent-transcripts/u.jsonl", cfg)
    assert should_skip_repo("PyRo1121/private", cfg)
    assert not should_skip_repo("other/public", cfg)


def test_parse_cursor_blob_adds_github_provenance() -> None:
    hit = _hit(".cursor/projects/p/agent-transcripts/uuid/uuid.jsonl")
    line = json.dumps(
        {
            "role": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Fix the failing test in auth module please."}
                ]
            },
        }
    )
    recs = list(parse_blob_text(hit, line + "\n", harness_hint="cursor", max_lines=100))
    assert len(recs) == 1
    assert recs[0]["source"] == "github_public"
    assert recs[0]["github_repo"] == "someone/agent-logs"
    assert recs[0]["github_path"] == hit.path
    assert recs[0]["harness"] == "cursor"


def test_looks_like_chat_antigravity_role_not_gemini_type() -> None:
    antigravity = json.dumps({"role": "user", "content": "hello from antigravity"})
    assert looks_like_chat_blob(antigravity, "antigravity", min_hits=1)
    gemini_type = json.dumps({"type": "user", "content": "hello from gemini"})
    assert looks_like_chat_blob(gemini_type, "gemini_cli", min_hits=1)
    assert not looks_like_chat_blob(antigravity, "gemini_cli", min_hits=1)


def test_parse_gemini_cli_jsonl_type_field() -> None:
    hit = _hit(".gemini/tmp/proj/chats/session-abc.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "user", "content": "How do I fix this test?"}),
            json.dumps({"type": "gemini", "content": "Run pytest with -q for quiet output."}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="gemini_cli", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"
    assert all(r["harness"] == "gemini_cli" for r in recs)
    assert all(r["source"] == "github_public" for r in recs)


def test_parse_gemini_cli_json_blob() -> None:
    hit = _hit(".gemini/tmp/proj/chats/session-abc.json")
    blob = json.dumps(
        {
            "messages": [
                {"type": "user", "content": "How do I run harvest tests?"},
                {"type": "gemini", "content": "Use uv run pytest with --cov."},
            ]
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="gemini_cli", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"
    assert all(r["harness"] == "gemini_cli" for r in recs)
    assert recs[0]["session_id"] == "session-abc"


def test_parse_antigravity_tokscale_jsonl() -> None:
    hit = _hit(".config/tokscale/antigravity-cache/sessions/sess.jsonl")
    blob = "\n".join(
        [
            json.dumps({"role": "user", "content": "Explain this module."}),
            json.dumps({"role": "assistant", "content": "It handles GitHub harvest parsing."}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="antigravity", max_lines=100))
    assert len(recs) == 2
    assert all(r["harness"] == "antigravity" for r in recs)


def test_parse_trae_tokscale_json() -> None:
    hit = _hit(".config/tokscale/trae-cache/sessions/sess.json")
    blob = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ]
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="trae", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "trae"
    assert recs[0]["role"] == "user"


def test_parse_qwen_cli_json_messages_bundle() -> None:
    hit = _hit(".qwen/projects/p/chats/session-1.json")
    blob = json.dumps(
        {
            "messages": [
                {"type": "user", "content": "Ping"},
                {"type": "model", "content": "Pong"},
            ]
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="qwen_cli", max_lines=100))
    assert len(recs) == 2
    assert recs[1]["role"] == "assistant"
    assert all(r["harness"] == "qwen_cli" for r in recs)


def test_parse_qwen_cli_jsonl_chat_record() -> None:
    hit = _hit(".qwen/projects/p/chats/uuid.jsonl")
    blob = "\n".join(
        [
            json.dumps(
                {
                    "type": "user",
                    "message": {"parts": [{"text": "Explain pytest"}]},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"parts": [{"text": "Run uv run pytest."}]},
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="qwen_cli", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["text"] == "Explain pytest"


def test_parse_qwen_stream_json_jsonl() -> None:
    hit = _hit(".qwen/projects/p/chats/uuid.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "session_start", "session_id": "s1"}),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Explain pytest"}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Run uv run pytest."}],
                    },
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="qwen_cli", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["text"] == "Explain pytest"
    assert recs[1]["role"] == "assistant"
    assert recs[1]["text"] == "Run uv run pytest."


def test_parse_kiro_cli_jsonl() -> None:
    hit = _hit(".kiro/sessions/cli/sess.jsonl")
    blob = "\n".join(
        [
            json.dumps({"role": "user", "content": "hello"}),
            json.dumps({"role": "assistant", "content": "hi there"}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="kiro", max_lines=100))
    assert len(recs) == 2


def test_parse_generic_jsonl() -> None:
    hit = _hit("transcripts/session.jsonl")
    blob = "\n".join(
        [
            json.dumps({"role": "user", "content": "How do I run pytest on this repo?"}),
            json.dumps({"role": "assistant", "content": "Use `uv run pytest packages/` from the repo root."}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="generic", max_lines=100))
    assert len(recs) == 2
    assert all(r["source"] == "github_public" for r in recs)


def test_looks_like_chat_blob_openclaw() -> None:
    lines = [
        json.dumps(
            {
                "type": "message",
                "message": {"role": "user", "content": "hello openclaw"},
            }
        ),
        json.dumps(
            {
                "type": "message",
                "message": {"role": "assistant", "content": "hi from claw"},
            }
        ),
    ]
    assert looks_like_chat_blob("\n".join(lines), "openclaw")
    assert not looks_like_chat_blob('{"type":"session_start"}\n{"type":"tool_call"}', "openclaw")


def test_parse_openclaw_blob() -> None:
    hit = _hit(".openclaw/agents/main/sessions/sess-1.jsonl")
    blob = "\n".join(
        [
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Run the harvest tests please."}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "custom_message",
                    "message": {"role": "assistant", "content": "Running pytest now."},
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="openclaw", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "openclaw"
    assert recs[0]["role"] == "user"
    assert "harvest tests" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert all(r["source"] == "github_public" for r in recs)


def test_parse_factory_blob_unwraps_messages() -> None:
    hit = _hit(".factory/projects/acme/session.jsonl")
    blob = "\n".join(
        [
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Fix the factory parser."},
                        {"role": "assistant", "content": "Added messages unwrap."},
                    ]
                }
            ),
            json.dumps({"role": "user", "content": "Direct line message."}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="factory", max_lines=100))
    assert len(recs) == 3
    assert all(r["harness"] == "factory" for r in recs)
    assert recs[0]["text"] == "Fix the factory parser."
    assert recs[2]["text"] == "Direct line message."


def test_parse_kimi_blob_list_content() -> None:
    hit = _hit(".kimi/sessions/uuid/context.jsonl")
    blob = "\n".join(
        [
            json.dumps({"role": "_system_prompt", "content": "ignore me"}),
            json.dumps(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Kimi list content block."}],
                }
            ),
            json.dumps({"role": "assistant", "content": "Parsed via kimi_sessions helper."}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="kimi", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "kimi"
    assert recs[0]["text"] == "Kimi list content block."
    assert recs[1]["text"] == "Parsed via kimi_sessions helper."


def test_parse_clawdbot_path_detects_openclaw() -> None:
    hit = _hit(".clawdbot/sessions/sess-2.jsonl")
    blob = json.dumps(
        {
            "type": "message",
            "message": {"role": "user", "content": "clawdbot legacy path"},
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint=None, max_lines=100))
    assert len(recs) == 1
    assert recs[0]["harness"] == "openclaw"
    assert recs[0]["text"] == "clawdbot legacy path"


def test_public_meta_defaults_github_public() -> None:
    rows = [{"source": "github_public", "harness": "cursor"}]
    meta = _public_meta_defaults(rows)
    assert meta["data_source"] == "public"
    assert meta["public_dataset"] == "github_public"


def test_load_harvest_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "github-harvest.yaml"
    cfg_file.write_text(
        """
github_harvest:
  max_files_per_run: 42
  queries:
    - id: test_q
      q: 'extension:jsonl fork:true'
""",
        encoding="utf-8",
    )
    cfg = load_harvest_config(cfg_file)
    assert cfg.max_files_per_run == 42
    assert len(cfg.queries) == 1
    assert cfg.queries[0]["id"] == "test_q"


def test_looks_like_chat_blob_rejects_fake_whole_json_messages() -> None:
    blob = json.dumps({"messages": [{"id": 1}, {"id": 2}]})
    assert not looks_like_chat_blob(blob, "continue")


def test_looks_like_openhands_single_event_json() -> None:
    blob = json.dumps(
        {
            "source": "user",
            "action": "message",
            "args": {"content": "Fix the failing test please"},
        }
    )
    assert looks_like_chat_blob(blob, "openhands", min_hits=2)


def test_looks_like_openhands_rejects_source_only() -> None:
    blob = json.dumps({"source": "agent", "kind": "ActionEvent"})
    assert not looks_like_chat_blob(blob, "openhands", min_hits=2)


def test_parse_openhands_event_json() -> None:
    hit = _hit(".openhands-state/sessions/sess-abc/events/42.json")
    blob = json.dumps(
        {
            "source": "user",
            "action": "message",
            "args": {"content": "Fix the failing test please"},
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="openhands", max_lines=100))
    assert len(recs) == 1
    assert recs[0]["harness"] == "openhands"
    assert recs[0]["role"] == "user"
    assert recs[0]["session_id"] == "sess-abc"
    assert "Fix the failing test" in recs[0]["text"]
    assert recs[0]["source"] == "github_public"


def test_looks_like_watchfire_type_human() -> None:
    lines = [
        json.dumps({"type": "human", "content": "How do I run pytest?"}),
        json.dumps({"type": "assistant", "content": "Use uv run pytest."}),
    ]
    assert looks_like_chat_blob("\n".join(lines), "watchfire")


def test_parse_watchfire_jsonl() -> None:
    hit = _hit(".watchfire/logs/session-1.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "human", "content": "How do I run pytest on this repo?"}),
            json.dumps(
                {"type": "assistant", "content": "Use `uv run pytest packages/` from the repo root."}
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="watchfire", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"
    assert recs[0]["harness"] == "watchfire"


_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "github_harvest"


def test_looks_like_chat_blob_continue_history_fixture() -> None:
    blob = (_FIXTURES / "continue_session.json").read_text(encoding="utf-8")
    assert looks_like_chat_blob(blob, "continue")


def test_looks_like_chat_blob_opencode_parts_fixture() -> None:
    blob = (_FIXTURES / "opencode_message.json").read_text(encoding="utf-8")
    assert looks_like_chat_blob(blob, "opencode")


def test_parse_continue_session_fixture() -> None:
    hit = _hit(".continue/sessions/fixture-continue-1.json")
    blob = (_FIXTURES / "continue_session.json").read_text(encoding="utf-8")
    recs = list(parse_blob_text(hit, blob, harness_hint="continue", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert "pytest" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert "uv run pytest" in recs[1]["text"]
    assert all(r["harness"] == "continue" for r in recs)


def test_parse_opencode_message_fixture() -> None:
    hit = _hit(".local/share/opencode/storage/message/sess/uuid.json")
    blob = (_FIXTURES / "opencode_message.json").read_text(encoding="utf-8")
    recs = list(parse_blob_text(hit, blob, harness_hint="opencode", max_lines=100))
    assert len(recs) == 1
    assert recs[0]["role"] == "assistant"
    assert "harvest parsers" in recs[0]["text"]
    assert "skip non-text" not in recs[0]["text"]


def test_parse_opencode_session_messages_fixture() -> None:
    hit = _hit("opencode/storage/session/sess-fixture.json")
    blob = (_FIXTURES / "opencode_session.json").read_text(encoding="utf-8")
    recs = list(parse_blob_text(hit, blob, harness_hint="opencode", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"


def test_parse_opencode_storage_part_json() -> None:
    hit = _hit(".local/share/opencode/storage/part/sess123/part-uuid.json")
    blob = json.dumps(
        {
            "type": "text",
            "role": "assistant",
            "text": "OpenCode split part text for harvest coverage.",
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="opencode", max_lines=100))
    assert len(recs) == 1
    assert recs[0]["role"] == "assistant"
    assert recs[0]["harness"] == "opencode"
    assert recs[0]["session_id"] == "sess123"
    assert "split part text" in recs[0]["text"]


def test_parse_aider_markdown_history() -> None:
    hit = _hit("project/.aider.chat.history.md")
    blob = (
        "# aider chat started\n\n"
        "#### user\n"
        "Raise github_harvest coverage with minimal tests.\n\n"
        "#### assistant\n"
        "Added gemini json, opencode part, and aider markdown tests.\n"
    )
    assert looks_like_chat_blob(blob, "aider")
    recs = list(parse_blob_text(hit, blob, harness_hint="aider", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"
    assert all(r["harness"] == "aider" for r in recs)


def test_parse_amp_thread_block_content() -> None:
    blob = json.dumps(
        {
            "id": "thread-fixture-1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the harvest parsers please."}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll add amp block-array content support."},
                        {"type": "tool_use", "name": "grep"},
                    ],
                },
            ],
        }
    )
    assert looks_like_chat_blob(blob, "amp")
    hit = _hit(".local/share/amp/threads/thread-fixture-1.json")
    recs = list(parse_blob_text(hit, blob, harness_hint="amp", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "amp"
    assert recs[0]["role"] == "user"
    assert "harvest parsers" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert "block-array content" in recs[1]["text"]
    assert all(r["source"] == "github_public" for r in recs)


def test_looks_like_chat_blob_cline_api_array_fixture() -> None:
    blob = (_FIXTURES / "cline_api_conversation_history.json").read_text(encoding="utf-8")
    assert looks_like_chat_blob(blob, "cline")
    assert looks_like_chat_blob(blob, "roo_code")
    assert not looks_like_chat_blob(json.dumps([{"id": 1}, {"id": 2}]), "cline")


def test_parse_cline_api_conversation_history_fixture() -> None:
    blob = (_FIXTURES / "cline_api_conversation_history.json").read_text(encoding="utf-8")
    hit = _hit(".cline/data/tasks/task-42/api_conversation_history.json")
    recs = list(parse_blob_text(hit, blob, harness_hint="cline", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "cline"
    assert recs[0]["role"] == "user"
    assert recs[0]["session_id"] == "task-42"
    assert recs[0]["source"] == "github_public"
    assert "login test" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert "auth module" in recs[1]["text"]


def test_parse_roo_api_conversation_history_fixture() -> None:
    blob = (_FIXTURES / "cline_api_conversation_history.json").read_text(encoding="utf-8")
    hit = _hit(
        "Code/User/globalStorage/rooveterinaryinc.roo-cline/tasks/abc123/"
        "api_conversation_history.json"
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="roo_code", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "roo_code"
    assert recs[0]["session_id"] == "abc123"


def test_parse_cline_ui_messages() -> None:
    blob = json.dumps(
        [
            {"role": "user", "text": "Run pytest on packages/dataprep"},
            {"role": "assistant", "text": "I'll run the test suite now."},
        ]
    )
    hit = _hit(".cline/data/tasks/t99/ui_messages.json")
    recs = list(parse_blob_text(hit, blob, harness_hint="cline", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["text"] == "Run pytest on packages/dataprep"
    assert recs[1]["role"] == "assistant"


def test_load_harvest_config_max_search_requests(tmp_path: Path) -> None:
    cfg_file = tmp_path / "github-harvest.yaml"
    cfg_file.write_text(
        """
github_harvest:
  max_search_requests_per_run: 500
  queries:
    - id: test_q
      q: 'extension:jsonl fork:true'
""",
        encoding="utf-8",
    )
    cfg = load_harvest_config(cfg_file)
    assert cfg.max_search_requests_per_run == 500


def test_load_harvest_config_rejects_bad_max_search_requests(tmp_path: Path) -> None:
    cfg_file = tmp_path / "github-harvest.yaml"
    cfg_file.write_text(
        """
github_harvest:
  max_search_requests_per_run: not-a-number
  queries:
    - id: test_q
      q: 'extension:jsonl fork:true'
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_harvest_config(cfg_file)


def test_run_harvest_dry_run_skips_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    for key in (
        "GITHUB_APP_CLIENT_ID",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_APP_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = HarvestConfig(
        max_files_per_run=5,
        state_path=tmp_path / "state.json",
        raw_prefix="public-github-sessions-test",
        queries=(
            {
                "id": "q1",
                "q": "extension:jsonl",
                "harness_hint": "generic",
                "require_path_substrings": ("transcripts/",),
            },
        ),
    )
    hit = _hit("transcripts/x.jsonl", qid="q1")
    fake_client = MagicMock()

    with patch("llm_dataprep.github_harvest.iter_code_search_hits", return_value=[hit]):
        with patch("llm_dataprep.github_harvest._github_token", return_value="tok"):
            with patch("llm_dataprep.github_harvest.GitHubClient", return_value=fake_client):
                stats = run_harvest(cfg, dry_run=True, reset_state=True)

    assert stats["files_fetched"] == 1
    fake_client.fetch_raw_file.assert_not_called()


def test_flush_pending_downloads_batch_rest_ingest(tmp_path: Path) -> None:
    cfg = HarvestConfig(
        state_path=tmp_path / "state.json",
        raw_prefix="public-github-sessions-test",
        download_mode="rest",
        max_file_bytes=1_000_000,
    )
    cache = HarvestCache(cfg.state_path)
    hit = _hit("misc/transcripts/chat.jsonl")
    key = f"{hit.repo_full_name}:{hit.path}:{hit.sha}"
    qspec = {"harness_hint": "generic"}
    pending = [(hit, key, qspec, "q1", "pat")]
    stats: dict[str, int] = {
        "files_fetched": 0,
        "files_skipped": 0,
        "records": 0,
        "files_rejected_content": 0,
        "rest_blobs": 0,
    }
    record_buf: list[dict] = []
    flushed: list[int] = []

    def flush_records() -> None:
        flushed.append(len(record_buf))
        record_buf.clear()

    client = MagicMock()
    client.fetch_file_bytes.return_value = b'{"role":"user","text":"hello from batch"}\n'

    _flush_pending_downloads_batch(
        pending,
        client=client,
        gql=None,
        cfg=cfg,
        cache=cache,
        stats=stats,
        record_buf=record_buf,
        flush_records=flush_records,
        dry_run=False,
    )

    assert stats["files_fetched"] == 1
    assert stats["records"] == 1
    assert stats["rest_blobs"] == 1
    assert flushed == [1]
    client.fetch_file_bytes.assert_called_once()


def test_run_harvest_ingests_downloaded_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    for key in (
        "GITHUB_APP_CLIENT_ID",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_APP_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = HarvestConfig(
        max_files_per_run=5,
        state_path=tmp_path / "state.json",
        raw_prefix="public-github-sessions-test",
        download_mode="rest",
        queries=(
            {
                "id": "q1",
                "q": "extension:jsonl",
                "harness_hint": "generic",
                "require_path_substrings": ("transcripts/",),
            },
        ),
    )
    hit = _hit("misc/transcripts/chat.jsonl", qid="q1")
    fake_client = MagicMock()
    fake_client.fetch_file_bytes.return_value = (
        b'{"role":"user","text":"harvested session line"}\n'
    )

    with patch("llm_dataprep.github_harvest.iter_code_search_hits", return_value=[hit]):
        with patch("llm_dataprep.github_harvest._github_token", return_value="tok"):
            with patch("llm_dataprep.github_harvest.GitHubClient", return_value=fake_client):
                with patch(
                    "llm_dataprep.github_harvest.append_records_buffered",
                    return_value=1,
                ) as append_mock:
                    stats = run_harvest(cfg, dry_run=False, reset_state=True)

    assert stats["files_fetched"] == 1
    assert stats["records"] == 1
    fake_client.fetch_file_bytes.assert_called_once()
    append_mock.assert_called()


def test_codex_event_msg_user() -> None:
    session_uuid = "019d4b4c-3972-72b2-888f-89d893b08a55"
    hit = _hit(
        f".codex/sessions/2026/04/01/rollout-2026-04-01T18-06-19-{session_uuid}.jsonl"
    )
    blob = "\n".join(
        [
            json.dumps({"type": "session_meta", "payload": {"id": session_uuid}}),
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "Why does npm install -g fail after hardening?",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Check npm prefix."}],
                    },
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="codex", max_lines=100))
    assert len(recs) == 2
    assert all(r["session_id"] == session_uuid for r in recs)
    assert recs[0]["role"] == "user"
    assert recs[0]["record_type"] == "event_msg"
    assert "npm install -g" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert looks_like_chat_blob(blob, "codex")


def test_codex_skips_duplicate_user_in_response_item_when_event_msg_present() -> None:
    session_uuid = "019d4b4c-3972-72b2-888f-89d893b08a55"
    hit = _hit(
        f".codex/sessions/2026/04/01/rollout-2026-04-01T18-06-19-{session_uuid}.jsonl"
    )
    user_text = "Same user turn in event_msg and response_item."
    blob = "\n".join(
        [
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": user_text},
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_text}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Only one assistant reply."}],
                    },
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="codex", max_lines=100))
    assert len(recs) == 2
    assert [r["role"] for r in recs] == ["user", "assistant"]
    assert recs[0]["record_type"] == "event_msg"
    assert recs[1]["text"] == "Only one assistant reply."


def test_pi_custom_message() -> None:
    hit = _hit(".pi/agent/sessions/--home-user-proj--/20260228_143022_abc123.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "session", "id": "abc123", "cwd": "/tmp"}),
            json.dumps(
                {
                    "type": "custom_message",
                    "content": "User pasted stack trace from CI failure.",
                    "display": True,
                }
            ),
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "I'll inspect the logs."}],
                    },
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="pi", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["role"] == "user"
    assert "stack trace" in recs[0]["text"]
    assert recs[0]["session_id"] == "abc123"
    assert looks_like_chat_blob(blob, "pi")


def test_copilot_sniff_rejects_generic_type_with_content() -> None:
    lines = [
        json.dumps({"type": "tool.call", "data": {"content": "not a chat turn"}}),
        json.dumps({"type": "session.start", "data": {"content": "also not chat"}}),
    ]
    assert not looks_like_chat_blob("\n".join(lines), "copilot")


def test_copilot_sniff_accepts_chronicle_message_types() -> None:
    lines = [
        json.dumps({"type": "user.message", "data": {"content": "Fix the harvest parser."}}),
        json.dumps({"type": "assistant.message", "data": {"content": "Updated copilot sniff."}}),
    ]
    assert looks_like_chat_blob("\n".join(lines), "copilot")


def test_copilot_vscode_sniff_accepts_kind2_patch() -> None:
    lines = [
        json.dumps({"kind": 0, "v": {"sessionId": "abc", "version": 3}}),
        json.dumps(
            {
                "kind": 2,
                "k": ["requests"],
                "v": [
                    {
                        "message": {"text": "How do I run pytest?"},
                        "response": [{"value": "Use uv run pytest."}],
                    }
                ],
            }
        ),
    ]
    assert looks_like_chat_blob("\n".join(lines), "copilot_vscode", min_hits=1)


def test_copilot_vscode_sniff_rejects_kind1_only() -> None:
    lines = [
        json.dumps({"kind": 0, "v": {"sessionId": "abc", "version": 3}}),
        json.dumps({"kind": 1, "k": ["customTitle"], "v": "My chat title"}),
    ]
    assert not looks_like_chat_blob("\n".join(lines), "copilot_vscode")


def test_parse_copilot_chronicle_blob() -> None:
    hit = _hit(".copilot/session-state/sess-1/events.jsonl")
    blob = "\n".join(
        [
            json.dumps({"type": "session.start", "data": {"content": "ignore"}}),
            json.dumps({"type": "user.message", "data": {"content": "Explain copilot harvest."}}),
            json.dumps({"type": "assistant.message", "data": {"content": "Chronicle events parser."}}),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="copilot", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "copilot"
    assert recs[0]["role"] == "user"
    assert recs[1]["role"] == "assistant"
    assert recs[0]["session_id"] == "sess-1"


def test_parse_copilot_vscode_jsonl_patch() -> None:
    hit = _hit(".config/Code/User/workspaceStorage/ws/chatSessions/sess-uuid.jsonl")
    blob = "\n".join(
        [
            json.dumps({"kind": 0, "v": {"sessionId": "sess-uuid", "version": 3}}),
            json.dumps({"kind": 1, "k": ["customTitle"], "v": "Harvest test"}),
            json.dumps(
                {
                    "kind": 2,
                    "k": ["requests"],
                    "v": [
                        {
                            "message": {"text": "Parse vscode copilot jsonl."},
                            "response": [{"value": "Kind 2 patch applied."}],
                        }
                    ],
                }
            ),
        ]
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="copilot_vscode", max_lines=100))
    assert len(recs) == 2
    assert recs[0]["harness"] == "copilot_vscode"
    assert recs[0]["role"] == "user"
    assert "vscode copilot" in recs[0]["text"]
    assert recs[1]["role"] == "assistant"
    assert recs[1]["text"] == "Kind 2 patch applied."
    assert recs[0]["session_id"] == "sess-uuid"
