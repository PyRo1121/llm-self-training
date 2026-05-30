"""Probe which local agent harnesses have on-disk data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_dataprep.harnesses import HARNESS_REGISTRY, HarnessSpec


@dataclass
class HarnessProbe:
    spec: HarnessSpec
    present: bool
    detail: str


def _count_glob(root: Path, pattern: str, limit: int = 5000) -> int:
    if not root.exists():
        return 0
    n = 0
    for _ in root.rglob(pattern):
        n += 1
        if n >= limit:
            break
    return n


def probe_harness(spec: HarnessSpec) -> HarnessProbe:
    hid = spec.harness_id
    root = spec.default_root

    if hid == "git":
        return HarnessProbe(spec, True, "requires --repo with .git")

    if hid == "cursor":
        n = _count_glob(root, "*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} jsonl transcripts" if n else "no transcripts")

    if hid == "codex":
        n = _count_glob(root, "rollout-*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} rollouts" if n else "no rollouts")

    if hid == "claude_code":
        n = _count_glob(root, "*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} jsonl" if n else "not installed")

    if hid == "pi":
        n = _count_glob(root, "*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} jsonl" if n else "no sessions")

    if hid == "opencode":
        db = Path.home() / ".local/share/opencode/opencode.db"
        if db.is_file():
            return HarnessProbe(spec, True, f"opencode.db ({db.stat().st_size // 1024} KB)")
        n = _count_glob(Path.home() / ".local/share/opencode/storage/message", "*.json")
        return HarnessProbe(spec, n > 0, f"legacy message json: {n}" if n else "no db")

    if hid == "t3code":
        db = root / "state.sqlite"
        return HarnessProbe(spec, db.is_file(), str(db) if db.is_file() else "no state.sqlite")

    if hid == "aider":
        n = 0
        for r in (Path.home() / "Documents", Path.home()):
            if r.exists():
                for _ in r.rglob(".aider.chat.history.md"):
                    n += 1
                    if n >= 20:
                        break
        return HarnessProbe(spec, n > 0, f"{n}+ markdown histories" if n else "none found")

    if hid == "cline":
        from llm_dataprep.paths_util import vscode_global_storage

        roots = vscode_global_storage("saoudrizwan.claude-dev")
        n = sum(1 for r in roots for _ in (r / "tasks").glob("*") if (r / "tasks").is_dir())
        cli = Path.home() / ".cline/data"
        present = bool(roots) or cli.is_dir()
        return HarnessProbe(spec, present, f"vscode={len(roots)} cli={cli.is_dir()} tasks~{n}")

    if hid == "continue":
        n = len(list(root.glob("*.json"))) if root.is_dir() else 0
        return HarnessProbe(spec, n > 0, f"{n} sessions" if n else "no sessions")

    if hid == "gemini_cli":
        n = _count_glob(root, "session-*.jsonl") + _count_glob(root, "session-*.json")
        return HarnessProbe(spec, n > 0, f"{n} session files" if n else "no chats")

    if hid == "copilot":
        n = sum(1 for d in root.iterdir() if d.is_dir() and (d / "events.jsonl").is_file()) if root.is_dir() else 0
        return HarnessProbe(spec, n > 0, f"{n} session dirs" if n else "no session-state")

    if hid == "amp":
        n = len(list(root.glob("*.json"))) if root.is_dir() else 0
        return HarnessProbe(spec, n > 0, f"{n} thread json" if n else "no threads")

    if hid == "factory":
        n = _count_glob(root / "projects", "*.jsonl") + _count_glob(root / "sessions", "*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} jsonl transcripts" if n else "no factory transcripts")

    if hid == "openhands":
        n = _count_glob(root, "*.json")
        return HarnessProbe(spec, n > 0, f"{n} event json" if n else "no openhands-state")

    if hid == "windsurf":
        from llm_dataprep.windsurf_vscdb import find_vscdb_files

        dbs = find_vscdb_files()
        pb = Path.home() / ".codeium/windsurf/cascade"
        pb_note = f"; {len(list(pb.glob('**/*')))} cascade files (encrypted .pb — not parsed)" if pb.is_dir() else ""
        return HarnessProbe(
            spec,
            bool(dbs),
            f"{len(dbs)} state.vscdb (partial ingest){pb_note}" if dbs else f"no Windsurf config{pb_note}",
        )

    if hid == "kimi":
        n = _count_glob(root, "context.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} context.jsonl" if n else "no sessions")

    if hid == "goose":
        db = root / "sessions.db"
        return HarnessProbe(spec, db.is_file(), str(db) if db.is_file() else "no sessions.db")

    if hid == "kiro":
        n = len(list(root.glob("*.json"))) + len(list(root.glob("*.jsonl"))) if root.is_dir() else 0
        sqlite = Path.home() / ".local/share/kiro-cli/data.sqlite3"
        present = n > 0 or sqlite.is_file()
        return HarnessProbe(spec, present, f"cli={n} sqlite={sqlite.is_file()}")

    if hid == "openclaw":
        n = _count_glob(root / "agents", "*.jsonl") if (root / "agents").is_dir() else 0
        return HarnessProbe(spec, n > 0, f"{n} jsonl" if n else "no agents/sessions")

    if hid == "zed_ai":
        db = root / "threads.db"
        return HarnessProbe(spec, db.is_file(), str(db) if db.is_file() else "no threads.db")

    if hid == "roo_code":
        from llm_dataprep.paths_util import vscode_global_storage

        roots = vscode_global_storage("rooveterinaryinc.roo-cline")
        n = sum(1 for r in roots for _ in (r / "tasks").glob("*") if (r / "tasks").is_dir())
        return HarnessProbe(spec, bool(roots), f"vscode roots={len(roots)} tasks~{n}")

    if hid == "mux":
        jsonl = _count_glob(root, "*.jsonl")
        usage = _count_glob(root, "session-usage.json")
        present = jsonl > 0 or usage > 0
        return HarnessProbe(
            spec,
            present,
            f"jsonl={jsonl} usage={usage}" if present else "no mux sessions",
        )

    if hid == "antigravity":
        cache = Path.home() / ".config/tokscale/antigravity-cache/sessions"
        n = len(list(cache.glob("*.jsonl"))) if cache.is_dir() else 0
        pb = Path.home() / ".gemini/antigravity/conversations"
        pb_n = len(list(pb.glob("*.pb"))) if pb.is_dir() else 0
        present = n > 0 or pb_n > 0
        detail = f"tokscale={n} jsonl"
        if pb_n:
            detail += f"; {pb_n} .pb (encrypted, not parsed)"
        return HarnessProbe(spec, present, detail if present else "run: tokscale antigravity sync")

    if hid == "trae":
        cache = Path.home() / ".config/tokscale/trae-cache/sessions"
        n = len(list(cache.glob("*.json"))) if cache.is_dir() else 0
        return HarnessProbe(spec, n > 0, f"{n} cached sessions" if n else "run: tokscale trae sync")

    if hid == "watchfire":
        n = _count_glob(root, "*.jsonl")
        return HarnessProbe(spec, n > 0, f"{n} jsonl logs" if n else "no logs")

    if hid == "crush":
        reg = root / "projects.json"
        n = 0
        for p in Path.home().rglob("crush.db"):
            if ".crush" in p.parts:
                n += 1
                if n >= 10:
                    break
        present = reg.is_file() or n > 0
        return HarnessProbe(spec, present, f"registry={reg.is_file()} dbs~{n}")

    if spec.ingest_tier == "detect":
        exists = root.is_dir()
        return HarnessProbe(spec, exists, "detect-only — parser not wired" if exists else "path missing")

    if root.exists():
        return HarnessProbe(spec, True, str(root))
    return HarnessProbe(spec, False, "path missing")


def probe_all() -> list[HarnessProbe]:
    return [probe_harness(h) for h in HARNESS_REGISTRY]
