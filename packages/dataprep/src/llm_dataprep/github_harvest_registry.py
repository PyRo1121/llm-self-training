"""GitHub code-search query registry for coding-agent session files.

Paths align with packages/dataprep/AGENT_HARNESSES.md and harnesses.py.
Each query uses: (1) GitHub `path:` / `filename:` in `q`, (2) post-hit path regex,
(3) parse gate in github_harvest.parse_blob_text (not content sniff alone).

Public GitHub reality (May 2026 probes): `.claude/projects` and `claude-session.jsonl`
yield real sessions; most other dotdirs (`.cursor/`, `.codex/`, `.pi/`) are almost
never committed — queries kept for accidental leaks and future growth.

Sources: pi-mono session-format docs, coding_agent_session_search, tokscale paths,
GitHub code-search path regex (docs.github.com search-github).
"""

from __future__ import annotations

from typing import Any

# Shared path rejects for benchmark / vendor noise (also in config/github-harvest.yaml)
DEFAULT_EXCLUDE_PATH_REGEX: tuple[str, ...] = (
    r"(?i)/jobs/",
    r"(?i)(?:^|/)harbor(?:/|$)",
    r"(?i)terminal_bench",
    r"(?i)swe-bench",
    r"(?i)/benchmark/",
    r"(?i)/evals/",
    r"(?i)/fixtures/",
    r"(?i)/subagents/",
    r"(?i)node_modules/",
)

# Cline / Roo VS Code storage noise when searching broad `.cline` paths
CLINE_CONFIG_REJECT: tuple[str, ...] = (
    r"(?i)(hooks|globalState|config|settings|mcp|state)\.json$",
)

HARVEST_QUERY_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "cursor_agent_transcripts",
        "harness_hint": "cursor",
        "label": "Cursor IDE agent-transcripts",
        "q": (
            '"role":"user" '
            r'path:/\.cursor\/projects\/ extension:jsonl fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.cursor/projects/[^/]+/agent-transcripts/[^/]+\.jsonl$",
            r"(?i)(?:^|/)\.cursor/projects/[^/]+/agent-transcripts/[^/]+/[^/]+\.jsonl$",
        ),
        "exclude_path_regex": (r"(?i)/subagents/",),
    },
    {
        "id": "codex_rollout_sessions",
        "harness_hint": "codex",
        "label": "OpenAI Codex CLI rollout",
        "q": (
            '"response_item" '
            r'path:/\.codex\/sessions\/ extension:jsonl fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.codex/sessions/.+/rollout-.+\.jsonl$",
        ),
    },
    {
        "id": "claude_code_projects",
        "harness_hint": "claude_code",
        "label": "Claude Code projects JSONL",
        "q": (
            '"type":"user" path:.claude/projects/ extension:jsonl fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.claude/projects/[^/]+/[^/]+\.jsonl$",
        ),
        "exclude_path_regex": (r"(?i)/subagents/",),
    },
    {
        "id": "claude_session_filename",
        "harness_hint": "claude_code",
        "label": "Claude Code claude-session.jsonl exports",
        "q": "filename:claude-session.jsonl extension:jsonl fork:true",
        "require_path_regex": (r"(?i)(?:^|/)claude-session\.jsonl$",),
    },
    {
        "id": "pi_agent_sessions",
        "harness_hint": "pi",
        "label": "Pi (pi-mono) agent sessions",
        "q": (
            '"type":"message" '
            r'path:/\.pi\/agent\/sessions\/ extension:jsonl fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.pi/agent/sessions/.+\.jsonl$",
        ),
    },
    {
        "id": "opencode_storage_session",
        "harness_hint": "opencode",
        "label": "OpenCode legacy storage/session JSON",
        "q": (
            '"role":"user" path:opencode/storage/session extension:json fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)opencode/storage/session/.+\.json$",
            r"(?i)(?:^|/)\.local/share/opencode/storage/session/.+\.json$",
        ),
    },
    {
        "id": "gemini_cli_chats",
        "harness_hint": "gemini_cli",
        "label": "Google Gemini CLI session chats",
        "q": (
            r'path:/\.gemini\/tmp\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.gemini/tmp/.+/chats/session-.+\.jsonl$",
        ),
    },
    {
        "id": "gemini_cli_chats_json",
        "harness_hint": "gemini_cli",
        "label": "Google Gemini CLI session chats (JSON)",
        "q": (
            r'path:/\.gemini\/tmp\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.gemini/tmp/.+/chats/session-.+\.json$",
        ),
    },
    {
        "id": "copilot_chronicle_events",
        "harness_hint": "copilot",
        "label": "GitHub Copilot CLI Chronicle events",
        "q": (
            r'path:/\.copilot\/session-state\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.copilot/session-state/[^/]+/events\.jsonl$",
        ),
    },
    {
        "id": "kimi_context_jsonl",
        "harness_hint": "kimi",
        "label": "Kimi CLI context.jsonl",
        "q": (
            r'path:/\.kimi\/sessions\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.kimi/sessions/.+/context\.jsonl$",
        ),
    },
    {
        "id": "factory_droid_projects",
        "harness_hint": "factory",
        "label": "Factory Droid project transcripts",
        "q": (
            r'path:/\.factory\/projects\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.factory/projects/.+\.jsonl$",
        ),
    },
    {
        "id": "openclaw_agent_sessions",
        "harness_hint": "openclaw",
        "label": "OpenClaw agent sessions",
        "q": (
            r'path:/\.openclaw\/agents\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.openclaw/agents/[^/]+/sessions/[^/]+\.jsonl$",
        ),
        "exclude_path_regex": (r"(?i)sessions\.jsonl$",),
    },
    {
        "id": "openhands_events",
        "harness_hint": "openhands",
        "label": "OpenHands session events JSON",
        "q": (
            r'path:/\.openhands-state\/sessions\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.openhands-state/sessions/[^/]+/events/.+\.json$",
            r"(?i)(?:^|/)conversations/.+/events/.+\.json$",
        ),
    },
    {
        "id": "amp_threads",
        "harness_hint": "amp",
        "label": "Amp (Sourcegraph) thread JSON",
        "q": (
            r'"role":"user" path:/\.local\/share\/amp\/threads\/ extension:json fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)amp/threads/[^/]+\.json$",
            r"(?i)(?:^|/)\.local/share/amp/threads/[^/]+\.json$",
        ),
    },
    {
        "id": "continue_sessions",
        "harness_hint": "continue",
        "label": "Continue.dev session JSON",
        "q": (
            r'path:/\.continue\/sessions\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.continue/sessions/[^/]+\.json$",
            r"(?i)(?:^|/)\.continue/projects/.+\.json$",
        ),
        "exclude_path_regex": (r"(?i)sessions\.json$",),
    },
    {
        "id": "cline_tasks",
        "harness_hint": "cline",
        "label": "Cline VS Code task JSON",
        "q": (
            '"role":"user" path:saoudrizwan.claude-dev/tasks extension:json fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.cline/data/tasks/.+\.json$",
            r"(?i)(?:^|/)saoudrizwan\.claude-dev/tasks/.+\.json$",
        ),
        "exclude_path_regex": CLINE_CONFIG_REJECT,
    },
    {
        "id": "roo_code_tasks",
        "harness_hint": "roo_code",
        "label": "Roo Code task JSON",
        "q": (
            '"role":"user" path:rooveterinaryinc.roo-cline/tasks extension:json fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)rooveterinaryinc\.roo-cline/tasks/.+\.json$",
        ),
        "exclude_path_regex": CLINE_CONFIG_REJECT,
    },
    {
        "id": "watchfire_logs",
        "harness_hint": "watchfire",
        "label": "Watchfire session JSONL logs",
        "q": (
            r'path:/\.watchfire\/logs\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.watchfire/logs/.+\.jsonl$",
        ),
    },
    {
        "id": "mux_sessions",
        "harness_hint": "mux",
        "label": "Mux session JSONL",
        "q": (
            r'path:/\.mux\/sessions\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.mux/sessions/.+\.jsonl$",
        ),
    },
    {
        "id": "kiro_cli_sessions",
        "harness_hint": "kiro",
        "label": "Kiro CLI sessions",
        "q": (
            r'path:/\.kiro\/sessions\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.kiro/sessions/cli/.+\.jsonl$",
        ),
    },
    {
        "id": "kiro_cli_sessions_json",
        "harness_hint": "kiro",
        "label": "Kiro CLI sessions (JSON)",
        "q": (
            r'path:/\.kiro\/sessions\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.kiro/sessions/cli/.+\.json$",
        ),
    },
    {
        "id": "antigravity_tokscale",
        "harness_hint": "antigravity",
        "label": "Antigravity tokscale session cache JSONL",
        "q": (
            r'path:/antigravity-cache\/sessions\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)antigravity-cache/sessions/.+\.jsonl$",
            r"(?i)(?:^|/)\.config/tokscale/antigravity-cache/sessions/.+\.jsonl$",
        ),
    },
    {
        "id": "trae_tokscale",
        "harness_hint": "trae",
        "label": "Trae tokscale session cache JSON",
        "q": (
            r'path:/trae-cache\/sessions\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)trae-cache/sessions/.+\.json$",
            r"(?i)(?:^|/)\.config/tokscale/trae-cache/sessions/.+\.json$",
        ),
    },
    {
        "id": "qwen_cli_chats",
        "harness_hint": "qwen_cli",
        "label": "Qwen Code CLI chat JSONL",
        "q": (
            "path:.qwen/projects extension:jsonl fork:true"
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.qwen/projects/.+/chats/.+\.jsonl$",
        ),
    },
    {
        "id": "vibe_mistral_messages",
        "harness_hint": "generic",
        "label": "Vibe (Mistral) messages JSONL",
        "q": (
            r'path:/\.vibe\/logs\/session\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.vibe/logs/session/.+/messages\.jsonl$",
        ),
    },
    {
        "id": "aider_chat_history",
        "harness_hint": "aider",
        "label": "Aider chat history markdown",
        "q": (
            "filename:aider.chat.history.md fork:true"
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.aider\.chat\.history\.md$",
            r"(?i)(?:^|/)aider\.chat\.history\.md$",
        ),
    },
    {
        "id": "clawdbot_sessions",
        "harness_hint": "openclaw",
        "label": "Clawdbot session JSONL",
        "q": (
            r'path:/\.clawdbot\/agents\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.clawdbot/agents/.+/sessions/.+\.jsonl$",
        ),
    },
    {
        "id": "copilot_vscode_chat",
        "harness_hint": "copilot_vscode",
        "label": "GitHub Copilot Chat VS Code chatSessions JSONL",
        "q": (
            "path:chatSessions extension:jsonl fork:true"
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)chatSessions/.+\.jsonl$",
        ),
    },
    {
        "id": "zed_threads_export",
        "harness_hint": "generic",
        "label": "Zed AI exported thread JSON (when committed)",
        "q": (
            r'path:/zed\/threads\/ extension:json fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)zed/threads/.+\.json$",
            r"(?i)(?:^|/)\.local/share/zed/threads/.+\.json$",
        ),
    },
    {
        "id": "goose_sessions_jsonl",
        "harness_hint": "generic",
        "label": "Goose (Block) legacy session JSONL",
        "q": (
            r'"role":"user" path:/goose/sessions extension:jsonl fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.local/share/goose/sessions/.+\.jsonl$",
            r"(?i)(?:^|/)\.config/goose/sessions/.+\.jsonl$",
        ),
    },
    {
        "id": "opencode_storage_part",
        "harness_hint": "opencode",
        "label": "OpenCode storage/part text JSON",
        "q": (
            '"type":"text" path:opencode/storage/part extension:json fork:true in:file'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)opencode/storage/part/.+\.json$",
            r"(?i)(?:^|/)\.local/share/opencode/storage/part/.+\.json$",
        ),
    },
    {
        "id": "opencode_storage_message",
        "harness_hint": "opencode",
        "label": "OpenCode legacy storage/message JSON",
        "q": (
            "path:opencode/storage/message extension:json fork:true"
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)opencode/storage/message/.+\.json$",
            r"(?i)(?:^|/)\.local/share/opencode/storage/message/.+\.json$",
        ),
    },
    {
        "id": "factory_droid_sessions",
        "harness_hint": "factory",
        "label": "Factory Droid sessions JSONL",
        "q": (
            r'path:/\.factory\/sessions\/ extension:jsonl fork:true'
        ),
        "require_path_regex": (
            r"(?i)(?:^|/)\.factory/sessions/.+\.jsonl$",
        ),
    },
)


# High-yield harnesses — 10 search pages (1000 hits) per warm run rotation
TIER_A_QUERY_IDS: frozenset[str] = frozenset(
    {
        "cursor_agent_transcripts",
        "codex_rollout_sessions",
        "claude_code_projects",
        "claude_session_filename",
        "pi_agent_sessions",
        "opencode_storage_session",
        "opencode_storage_message",
        "gemini_cli_chats",
        "copilot_chronicle_events",
        "kimi_context_jsonl",
        "factory_droid_projects",
        "continue_sessions",
        "cline_tasks",
        "roo_code_tasks",
        "aider_chat_history",
    }
)


def registry_queries(
    *,
    enabled: tuple[str, ...] | None = None,
    disabled: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return registry queries, optionally filtered by id."""
    disabled_set = frozenset(disabled or ())
    out: list[dict[str, Any]] = []
    for spec in HARVEST_QUERY_REGISTRY:
        qid = spec["id"]
        if qid in disabled_set:
            continue
        if enabled is not None and qid not in enabled:
            continue
        row = dict(spec)
        if qid in TIER_A_QUERY_IDS:
            row.setdefault("priority", 10)
            row.setdefault("max_pages", 10)
        else:
            row.setdefault("priority", 50)
        out.append(row)
    out.sort(key=lambda s: (int(s.get("priority", 50)), str(s["id"])))
    return tuple(out)


def registry_query_ids() -> tuple[str, ...]:
    return tuple(q["id"] for q in HARVEST_QUERY_REGISTRY)
