"""Resolve platform-specific agent log roots (Linux defaults, May 2026)."""

from __future__ import annotations

import os
from pathlib import Path


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def vscode_global_storage(extension_id: str) -> list[Path]:
    """Cline / similar extensions under VS Code family editors."""
    home = Path.home()
    candidates = [
        home / ".config/Code/User/globalStorage" / extension_id,
        home / ".config/Cursor/User/globalStorage" / extension_id,
        home / ".config/VSCodium/User/globalStorage" / extension_id,
        home / ".config/Code - OSS/User/globalStorage" / extension_id,
    ]
    return [p for p in candidates if p.is_dir()]


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None
