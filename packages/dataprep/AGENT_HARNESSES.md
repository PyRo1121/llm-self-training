# Local agent harness catalog (May 2026)

Reference for `llm-dataprep` ingest. Run `uv run --package llm-dataprep agent-ingest --list-harnesses` to probe **your** machine.

## Ingest tiers

| Tier | Meaning |
|------|---------|
| **full** | Parser in `agent-ingest`; plaintext local logs |
| **partial** | Some data recoverable; gaps documented |
| **detect** | Path registered; parser not implemented yet |
| **blocked** | No reliable local plaintext (encryption / server-only) |

## Windsurf — can you decrypt `.pb` files?

**No — not in any practical, supported way.**

- `~/.codeium/windsurf/cascade/` and `implicit/` store **per-UUID AES-encrypted** `.pb` blobs ([Codeium #127](https://github.com/Exafunction/codeium/issues/127)). Keys are not on disk; swapping files between UUIDs fails.
- Much history is **server-side**; local cache is incomplete and auto-pruned (~20 sessions).
- **What works instead:**
  1. **UI export** — Cascade ⋮ → Download Trajectory (markdown).
  2. **SQLite `state.vscdb`** — Chat/Cascade bubbles in `ItemTable` (JSON), similar to VS Code. We ingest this as harness `windsurf` (**partial**). See [0xSero/ai-data-extraction](https://github.com/0xSero/ai-data-extraction/blob/main/extract_windsurf.py), [windsurf-trajectory-extractor](https://github.com/jijiamoer/windsurf-trajectory-extractor) (protobuf in vscdb; may break on updates).
  3. **Do not** expect reverse-engineering `.pb` for training — ToS/risk and keys unavailable.

## Harness table

| ID | Tier | Default path(s) | Notes |
|----|------|-----------------|-------|
| cursor | full | `~/.cursor/projects/*/agent-transcripts/` | Exclude `subagents/` default |
| codex | full | `~/.codex/sessions/**/rollout-*.jsonl` | `--max-codex-mb` |
| claude_code | full | `~/.claude/projects/<hash>/*.jsonl` | |
| pi | full | `~/.pi/agent/sessions/` | `PI_CODING_AGENT_SESSION_DIR` |
| opencode | full | `~/.local/share/opencode/opencode.db` | Legacy `storage/message/*.json` |
| t3code | full | `~/.t3/userdata/state.sqlite` | Not t3.chat web export |
| aider | full | `**/.aider.chat.history.md` | Scans `~/Documents`, `~` |
| cline | full | VS Code `globalStorage/saoudrizwan.claude-dev/tasks` | + `~/.cline/data` |
| continue | full | `~/.continue/sessions/*.json` | |
| gemini_cli | full | `~/.gemini/tmp/*/chats/session-*.jsonl` | `GEMINI_DIR` |
| copilot | full | `~/.copilot/session-state/*/events.jsonl` | [Chronicle docs](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/chronicle) |
| amp | full | `~/.local/share/amp/threads/*.json` | Rich thread JSON + usage |
| factory | full | `~/.factory/projects/**/*.jsonl` | Droid transcripts |
| openhands | full | `~/.openhands-state/sessions/*/events/*.json` | Or `file_store_path` from config |
| windsurf | partial | `~/.config/Windsurf/User/**/state.vscdb` | **Not** `.pb` decrypt |
| git | full | `--repo` | PyDriller |
| kimi | full | `~/.kimi/sessions/**/context.jsonl` | [data-locations](https://github.com/MoonshotAI/kimi-cli/blob/main/docs/en/configuration/data-locations.md) |
| mux | partial | `~/.mux/sessions/` | `*.jsonl` if present; else `session-usage.json` metadata |
| goose | full | `~/.local/share/goose/sessions/sessions.db` | `GOOSE_PATH_ROOT` |
| kiro | full | `~/.kiro/sessions/cli/` | + `~/.local/share/kiro-cli/data.sqlite3` |
| openclaw | full | `~/.openclaw/agents/*/sessions/*.jsonl` | |
| zed_ai | full | `~/.local/share/zed/threads/threads.db` | `uv sync --extra zed` for zstd |
| roo_code | full | VS Code `rooveterinaryinc.roo-cline` | Same task layout as Cline |
| jetbrains_ai | detect | `~/.config/JetBrains` | IDE-specific — not wired |
| antigravity | partial | `~/.config/tokscale/antigravity-cache/sessions/*.jsonl` | `tokscale antigravity sync`; **not** `~/.gemini/antigravity/*.pb` |
| trae | partial | `~/.config/tokscale/trae-cache/sessions/*.json` | `tokscale trae sync` |
| watchfire | full | `~/.watchfire/logs/**/*.jsonl` | [daemon docs](https://watchfire.io/docs/components/daemon) |
| crush | full | `~/.local/share/crush/projects.json` → `*/.crush/crush.db` | |
| hermes | detect | app-specific | |

## Adding a harness

1. Add row to `harnesses.py` (`HarnessSpec` + `ingest_tier`).
2. Implement `packages/dataprep/src/llm_dataprep/<name>.py`.
3. Wire `agent_ingest.py` + `discover.py`.
4. Document path here and in `PLAN.md`.

## Sources used for paths

- [coding_agent_session_search](https://github.com/Dicklesworthstone/coding_agent_session_search) plan
- [tokscale](https://github.com/junhoyeo/tokscale) client path table
- Official docs: Pi, OpenCode, Gemini CLI, Copilot Chronicle, Factory hooks, OpenHands persistence
