"""Registry of local coding-agent harnesses (see packages/dataprep/AGENT_HARNESSES.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

IngestTier = str  # full | partial | detect | blocked


@dataclass(frozen=True)
class HarnessSpec:
    harness_id: str
    label: str
    default_root: Path
    format: str
    ingest_tier: IngestTier = "full"
    docs: str = ""
    notes: str = ""
    env_override: str = ""


def _pi_sessions_root() -> Path:
    base = os.environ.get("PI_CODING_AGENT_SESSION_DIR") or os.environ.get("PI_AGENT_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".pi/agent/sessions"


def _t3_home() -> Path:
    return Path(os.environ.get("T3CODE_HOME", os.environ.get("T3_HOME", "~/.t3"))).expanduser()


def _gemini_root() -> Path:
    return Path(os.environ.get("GEMINI_DIR", "~/.gemini")).expanduser() / "tmp"


def _windsurf_config_root() -> Path:
    return Path.home() / ".config/Windsurf"


def _detect_only(
    harness_id: str,
    label: str,
    root: Path,
    fmt: str,
    notes: str,
    docs: str = "",
) -> HarnessSpec:
    return HarnessSpec(
        harness_id,
        label,
        root,
        fmt,
        ingest_tier="detect",
        docs=docs,
        notes=notes,
    )


HARNESS_REGISTRY: tuple[HarnessSpec, ...] = (
    # --- full ingest ---
    HarnessSpec("cursor", "Cursor IDE", Path.home() / ".cursor/projects", "jsonl", notes="agent-transcripts; exclude subagents/ default", docs="https://github.com/S2thend/cursor-history"),
    HarnessSpec("codex", "OpenAI Codex CLI", Path.home() / ".codex/sessions", "jsonl", notes="rollout-*.jsonl", env_override="CODEX_HOME"),
    HarnessSpec("claude_code", "Claude Code", Path.home() / ".claude/projects", "jsonl", docs="https://code.claude.com/docs"),
    HarnessSpec("pi", "Pi (pi-mono)", _pi_sessions_root(), "jsonl", env_override="PI_CODING_AGENT_SESSION_DIR", docs="https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/sessions.md"),
    HarnessSpec("opencode", "OpenCode", Path.home() / ".local/share/opencode", "sqlite", env_override="OPENCODE_DATA_DIR", docs="https://opencode.ai/docs/troubleshooting/"),
    HarnessSpec("t3code", "T3 Code", _t3_home() / "userdata", "sqlite", env_override="T3CODE_HOME"),
    HarnessSpec("aider", "Aider", Path.home(), "markdown", env_override="AIDER_CHAT_HISTORY_FILE", docs="https://aider.chat/docs/config/options.html"),
    HarnessSpec("cline", "Cline", Path.home() / ".cline/data", "json", docs="https://github.com/cline/cline/issues/7742"),
    HarnessSpec("continue", "Continue.dev", Path.home() / ".continue/sessions", "json", env_override="CONTINUE_GLOBAL_DIR"),
    HarnessSpec("gemini_cli", "Gemini CLI", _gemini_root(), "jsonl", env_override="GEMINI_DIR"),
    HarnessSpec("copilot", "GitHub Copilot CLI", Path.home() / ".copilot/session-state", "jsonl", docs="https://docs.github.com/en/copilot/concepts/agents/copilot-cli/chronicle", notes="events.jsonl per session dir"),
    HarnessSpec("amp", "Amp (Sourcegraph)", Path.home() / ".local/share/amp/threads", "json", docs="https://ampcode.com/manual", notes="thread JSON under threads/"),
    HarnessSpec("factory", "Factory Droid", Path.home() / ".factory", "jsonl", docs="https://docs.factory.ai/reference/hooks-reference", notes="~/.factory/projects/**/*.jsonl transcripts"),
    HarnessSpec("openhands", "OpenHands", Path.home() / ".openhands-state", "json", docs="https://docs.openhands.dev/sdk/guides/convo-persistence", notes="sessions/*/events/*.json"),
    HarnessSpec("git", "Git (PyDriller)", Path.cwd(), "git", docs="https://pydriller.readthedocs.io/", notes="--repo required"),
    # --- partial ---
    HarnessSpec(
        "windsurf",
        "Windsurf (state.vscdb)",
        _windsurf_config_root(),
        "sqlite",
        ingest_tier="partial",
        docs="https://github.com/Exafunction/codeium/issues/127",
        notes=".pb AES cache NOT decrypted; reads ItemTable JSON from state.vscdb only",
    ),
    # --- full (detect tier wired) ---
    HarnessSpec("kimi", "Kimi CLI", Path.home() / ".kimi/sessions", "jsonl", docs="https://github.com/MoonshotAI/kimi-cli/blob/main/docs/en/configuration/data-locations.md", notes="context.jsonl per session"),
    HarnessSpec("goose", "Goose", Path.home() / ".local/share/goose/sessions", "sqlite", env_override="GOOSE_PATH_ROOT", notes="sessions.db messages table"),
    HarnessSpec("kiro", "Kiro CLI", Path.home() / ".kiro/sessions/cli", "json", docs="https://kiro.dev/docs/cli/chat/session-management/", notes="JSON/JSONL + kiro-cli data.sqlite3"),
    HarnessSpec("openclaw", "OpenClaw", Path.home() / ".openclaw", "jsonl", notes="agents/*/sessions/*.jsonl"),
    HarnessSpec("zed_ai", "Zed AI", Path.home() / ".local/share/zed/threads", "sqlite", notes="threads.db zstd blobs; pip install zstandard for decompress"),
    HarnessSpec("roo_code", "Roo Code", Path.home() / ".config/Code/User/globalStorage", "json", notes="rooveterinaryinc.roo-cline tasks/"),
    HarnessSpec("watchfire", "Watchfire proxy", Path.home() / ".watchfire/logs", "jsonl", docs="https://watchfire.io/docs/components/daemon", notes="session *.jsonl transcripts"),
    HarnessSpec("crush", "Crush", Path.home() / ".local/share/crush", "sqlite", docs="https://github.com/charmbracelet/crush", notes="projects.json → */.crush/crush.db"),
    # --- partial ---
    HarnessSpec(
        "mux",
        "Mux",
        Path.home() / ".mux/sessions",
        "json",
        ingest_tier="partial",
        notes="*.jsonl transcripts if present; else session-usage.json metadata only",
    ),
    HarnessSpec(
        "antigravity",
        "Google Antigravity",
        Path.home() / ".config/tokscale/antigravity-cache/sessions",
        "jsonl",
        ingest_tier="partial",
        docs="https://github.com/junhoyeo/tokscale",
        notes="tokscale antigravity sync cache; ~/.gemini/antigravity/*.pb NOT decrypted",
    ),
    HarnessSpec(
        "trae",
        "Trae IDE",
        Path.home() / ".config/tokscale/trae-cache/sessions",
        "json",
        ingest_tier="partial",
        docs="https://github.com/junhoyeo/tokscale",
        notes="tokscale trae sync cache",
    ),
    _detect_only("jetbrains_ai", "JetBrains AI", Path.home() / ".config/JetBrains", "sqlite", "IDE-specific DB layout — not wired"),
)


def harnesses_for_ingest(tier: IngestTier | None = None) -> tuple[HarnessSpec, ...]:
    if tier is None:
        return tuple(h for h in HARNESS_REGISTRY if h.ingest_tier in ("full", "partial"))
    return tuple(h for h in HARNESS_REGISTRY if h.ingest_tier == tier)


def get_harness(harness_id: str) -> HarnessSpec:
    for h in HARNESS_REGISTRY:
        if h.harness_id == harness_id:
            return h
    raise KeyError(f"Unknown harness: {harness_id}")
