"""Hugging Face dataset loaders → raw JSONL records."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.public.records import (
    _message_text,
    iter_messages_records,
    make_record,
)

logger = logging.getLogger(__name__)

_LOADER_CTX: dict[str, Any] = {"local_dir": None, "shard_files": None}


def set_loader_context(
    *,
    local_dir: str | Path | None,
    shard_files: list[str | Path] | None = None,
) -> None:
    _LOADER_CTX["local_dir"] = str(local_dir) if local_dir else None
    _LOADER_CTX["shard_files"] = (
        [str(p) for p in shard_files] if shard_files else None
    )


def _active_local_dir(local_dir: str | Path | None) -> Path | None:
    if local_dir is not None:
        root = Path(local_dir)
    else:
        raw = _LOADER_CTX.get("local_dir")
        root = Path(raw) if raw else None
    if root is None or not root.is_dir():
        return None
    return root


def _active_shard_files(
    shard_files: list[str | Path] | None,
    root: Path,
) -> list[Path]:
    if shard_files is not None:
        return [Path(p) for p in shard_files]
    ctx = _LOADER_CTX.get("shard_files")
    if ctx:
        return [Path(p) for p in ctx]
    from llm_dataprep.public.hf_cache import list_parquet_shards

    return list_parquet_shards(root)

_CODE_HINTS = frozenset(
    {
        "code",
        "coding",
        "program",
        "debug",
        "software",
        "swe",
        "agent",
        "tool",
        "python",
        "rust",
        "typescript",
        "javascript",
        "go",
        "java",
    }
)


def _hf_token() -> str | None:
    """HF token: env vars first, then `hf auth login` cached token."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _hf_auth_label() -> str:
    token = _hf_token()
    if not token:
        return "not authenticated (run: hf auth login)"
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=token).whoami()
        name = who.get("name") or who.get("fullname") or "unknown"
        return f"authenticated as {name}"
    except Exception:
        return "authenticated (token present)"


def _iter_parquet_file(path: Path, *, batch_size: int = 256) -> Iterator[dict[str, Any]]:
    """Memory-mapped parquet row iteration (no datasets Arrow cache rebuild)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _iter_local_parquet_shards(shard_paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in shard_paths:
        yield from _iter_parquet_file(path)


def _stream_dataset(
    repo: str,
    *,
    split: str = "train",
    data_files: str | dict[str, str] | list[str] | None = None,
    config: str | None = None,
    token: str | None = None,
    local_dir: str | Path | None = None,
    shard_files: list[str | Path] | None = None,
    streaming: bool | None = None,
):
    """Load rows from a HF dataset — local cache (mmap) or Hub (streaming).

    When a local HF snapshot cache is active (via ``set_loader_context`` or
    ``local_dir``), parquet/json files are read from disk with
    ``streaming=False`` so datasets uses memory-mapped Arrow/Parquet (fast).
    Remote Hub access uses ``streaming=True`` to avoid re-downloading per row.
    """
    from datasets import load_dataset

    root = _active_local_dir(local_dir)
    use_streaming = streaming if streaming is not None else root is None

    if root is not None:
        if data_files is not None and not isinstance(data_files, dict):
            rel = data_files if isinstance(data_files, str) else data_files[0]
            path = root / rel
            if path.is_file():
                if path.suffix == ".parquet":
                    return _iter_local_parquet_shards([path])
                return load_dataset(
                    "json",
                    data_files=str(path),
                    split=split,
                    streaming=False,
                )
        shard_paths = _active_shard_files(shard_files, root)
        if shard_paths:
            return _iter_local_parquet_shards(shard_paths)
        try:
            kwargs: dict[str, Any] = {
                "path": str(root),
                "split": split,
                "streaming": False,
            }
            if config:
                kwargs["name"] = config
            if token:
                kwargs["token"] = token
            return load_dataset(**kwargs)
        except Exception as exc:
            logger.error(
                "Failed to load local HF cache at %s (split=%s, config=%s): %s",
                root,
                split,
                config,
                exc,
            )
            raise

    kwargs: dict[str, Any] = {"split": split, "streaming": use_streaming}
    if data_files:
        kwargs["data_files"] = data_files
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    return load_dataset(repo, **kwargs)


def _stream_hf_jsonl(
    repo: str,
    relpath: str,
    *,
    token: str | None = None,
    local_dir: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream JSONL from a local HF cache copy or remote repo."""
    root = _active_local_dir(local_dir)
    if root is not None:
        path = root / relpath
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    path = f"datasets/{repo}/{relpath}"
    with fs.open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            nested = val.get("text") or val.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _parse_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _row_looks_code_related(row: dict[str, Any]) -> bool:
    """Heuristic filter for mixed mega-datasets (e.g. UltraData code subset)."""
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("domain", "category", "task", "task_type", "source", "tags", "type", "skill")
    ).lower()
    if not blob.strip():
        return True
    return any(h in blob for h in _CODE_HINTS)


def _load_instruction_pairs(
    ds: Iterator[dict[str, Any]],
    *,
    dataset_id: str,
    hf_repo: str,
    max_rows: int | None,
    verify: str,
    user_keys: tuple[str, ...] = (
        "instruction",
        "prompt",
        "input",
        "question",
        "problem",
        "query",
    ),
    assistant_keys: tuple[str, ...] = (
        "response",
        "output",
        "answer",
        "solution",
        "completion",
        "code",
    ),
    thinking_keys: tuple[str, ...] = ("thinking", "reasoning", "chain_of_thought", "cot"),
    row_filter: Any | None = None,
    extra_fn: Any | None = None,
) -> Iterator[dict[str, Any]]:
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        if row_filter and not row_filter(row):
            continue
        user = _first_text(row, *user_keys)
        assistant = _first_text(row, *assistant_keys)
        thinking = _first_text(row, *thinking_keys)
        if thinking and assistant:
            assistant = f"[thinking]\n{thinking}\n\n{assistant}"
        if not user or not assistant:
            messages = _parse_messages(row.get("messages") or row.get("conversations"))
            if messages:
                sid = str(
                    row.get("id")
                    or row.get("session_id")
                    or row.get("trajectory_id")
                    or row.get("instance_id")
                    or f"{dataset_id}-{idx}"
                )
                extra = extra_fn(row) if extra_fn else None
                yield from iter_messages_records(
                    dataset_id=dataset_id,
                    session_id=sid,
                    messages=messages,
                    hf_repo=hf_repo,
                    map_tool_to_user=True,
                    verify=verify,
                    extra=extra,
                )
                count += 1
            continue
        sid = str(row.get("id") or row.get("sample_id") or row.get("fingerprint") or f"{dataset_id}-{idx}")
        extra = extra_fn(row) if extra_fn else None
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=1,
            role="user",
            text=user,
            hf_repo=hf_repo,
            verify=verify,
            extra=extra,
        )
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=2,
            role="assistant",
            text=assistant,
            hf_repo=hf_repo,
            verify=verify,
            extra=extra,
        )
        count += 1


def _require_gated_token(hf_repo: str) -> str:
    token = _hf_token()
    if not token:
        raise RuntimeError(
            f"{hf_repo} is gated. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
            f"and accept access at https://huggingface.co/datasets/{hf_repo}"
        )
    return token


def load_swe_next(
    *,
    max_rows: int | None = None,
    hf_repo: str = "TIGER-Lab/SWE-Next-SFT-Trajectories",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """PLAN: map tool→user; exec verified trajectories."""
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        sid = row.get("id") or row.get("instance_id") or f"swe-next-{idx}"
        yield from iter_messages_records(
            dataset_id="swe_next",
            session_id=str(sid),
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=True,
            verify="swe_next_verified",
        )
        count += 1


def load_swe_chat(
    *,
    max_rows: int | None = None,
    hf_repo: str = "SALT-NLP/SWE-chat",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    token = _hf_token()
    if not token:
        raise RuntimeError(
            f"{hf_repo} is gated. Run `hf auth login`, accept terms on the dataset card, "
            f"or set HF_TOKEN. See https://huggingface.co/datasets/{hf_repo}"
        )
    ds = _stream_dataset(hf_repo, data_files="conversations.parquet", token=token)
    emitted = 0
    current_sid: str | None = None
    turns: list[dict[str, Any]] = []

    def _emit_session(sid: str, session_turns: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        if not session_turns:
            return
        session_turns.sort(key=lambda t: t.get("turn", 0))
        messages = [{"role": t["role"], "content": t["content"]} for t in session_turns]
        yield from iter_messages_records(
            dataset_id="swe_chat",
            session_id=sid,
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=False,
            verify="swe_chat_wild",
            extra={"public_release": "2026-04"},
        )

    for row in ds:
        if max_rows is not None and emitted >= max_rows:
            break
        sid = str(row.get("session_id") or "unknown")
        role = (row.get("role") or "").lower()
        turn_type = (row.get("turn_type") or "").lower()
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if role == "metadata":
            continue
        if turn_type in ("assistant_thinking", "progress", "system_event", "file_snapshot"):
            continue
        out_role = role
        prefix = ""
        if role == "tool_use":
            out_role = "user"
            prefix = f"[tool:{row.get('tool_name') or 'call'}]\n"
        elif role == "tool_result":
            out_role = "user"
            prefix = "[tool_result]\n"
        elif role not in ("user", "assistant"):
            continue
        if sid != current_sid:
            if current_sid is not None:
                yield from _emit_session(current_sid, turns)
            current_sid = sid
            turns = []
        turns.append(
            {"role": out_role, "content": prefix + content, "turn": row.get("turn_number", 0)}
        )
        emitted += 1

    if current_sid is not None:
        yield from _emit_session(current_sid, turns)


def _append_tool_calls(content: str, tool_calls: Any) -> str:
    if not tool_calls or not isinstance(tool_calls, list):
        return content
    parts = [content] if content else []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or "tool"
        args = fn.get("arguments") or ""
        text = str(args)
        if len(text) > 480:
            text = text[:477] + "…"
        parts.append(f"[tool {name}] {text}")
    return "\n".join(p for p in parts if p).strip()


def _normalize_openhands_messages(trajectory: list[Any]) -> list[dict[str, Any]]:
    """OpenHands SWE-Zero rows: role/content (+ optional tool_calls on assistant)."""
    out: list[dict[str, Any]] = []
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").lower()
        content = _message_text(item.get("content"))
        if role == "assistant":
            content = _append_tool_calls(content, item.get("tool_calls"))
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def load_swe_zero_12m(
    *,
    max_rows: int | None = None,
    hf_repo: str = "AlienKevin/SWE-ZERO-12M-trajectories",
    exit_status: str | None = "Submitted",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """SWE-Zero 12M mini-swe-agent trajectories (messages[] per row)."""
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        status = row.get("exit_status")
        if exit_status is not None and status != exit_status:
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        iid = row.get("instance_id") or f"swe-zero-12m-{idx}"
        sid = f"{iid}-{idx}"
        extra = {
            "repo": row.get("repo"),
            "exit_status": status,
            "trajectory_format": row.get("trajectory_format"),
            "duration_sec": row.get("duration_sec"),
        }
        yield from iter_messages_records(
            dataset_id="swe_zero_12m",
            session_id=str(sid),
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=True,
            verify="swe_zero_trajectory",
            extra=extra,
        )
        count += 1


def load_swe_zero_openhands(
    *,
    max_rows: int | None = None,
    hf_repo: str = "nvidia/SWE-Zero-openhands-trajectories",
    require_patch: bool = True,
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """NVIDIA SWE-Zero OpenHands trajectories (trajectory[] + model_patch)."""
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        patch = (row.get("model_patch") or "").strip()
        if require_patch and not patch:
            continue
        traj = row.get("trajectory")
        if not isinstance(traj, list) or len(traj) < 2:
            continue
        messages = _normalize_openhands_messages(traj)
        if len(messages) < 2:
            continue
        sid = row.get("trajectory_id") or row.get("instance_id") or f"swe-zero-oh-{idx}"
        extra = {
            "repo": row.get("repo"),
            "instance_id": row.get("instance_id"),
            "source_dataset": row.get("dataset"),
            "license": row.get("license"),
            "has_patch": bool(patch),
        }
        yield from iter_messages_records(
            dataset_id="swe_zero_openhands",
            session_id=str(sid),
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=True,
            verify="swe_zero_openhands",
            extra=extra,
        )
        count += 1


def load_opencode_refined(
    *,
    max_rows: int | None = None,
    hf_repo: str = "EER6/nvidia-OpenCodeInstruct-refined",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    return _load_opencode_pair(hf_repo=hf_repo, dataset_id="opencode_refined", max_rows=max_rows)


def _load_opencode_pair(
    *,
    hf_repo: str,
    dataset_id: str,
    max_rows: int | None,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if not inp or not out:
            continue
        sid = str(row.get("id") or f"{dataset_id}-{idx}")
        score = row.get("average_test_score")
        extra = {"average_test_score": score, "domain": row.get("domain")}
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=1,
            role="user",
            text=inp,
            hf_repo=hf_repo,
            verify="unit_tests",
            extra=extra,
        )
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=2,
            role="assistant",
            text=out,
            hf_repo=hf_repo,
            verify="unit_tests",
            extra=extra,
        )
        count += 1


def _parse_messages_field(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict)]
    return []


def load_zen_agentic(**_: Any) -> Iterator[dict[str, Any]]:
    """zenlm/zen-agentic-dataset ships zstd JSONL shards; HF repo is placeholder until access."""
    raise RuntimeError(
        "zenlm/zen-agentic-dataset has no streamable public files on HF yet. "
        "Request access at oss@hanzo.ai (see dataset card). "
        "When shards are available, install llm-dataprep[zed] for zstd streaming."
    )
    yield {}  # pragma: no cover


def load_agentic_sft_new(
    *,
    max_rows: int | None = None,
    hf_repo: str = "WaltonFuture/agentic-sft-new",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    yield from _load_instruction_pairs(
        ds,
        dataset_id="agentic_sft_new",
        hf_repo=hf_repo,
        max_rows=max_rows,
        verify="agentic_sft",
    )


def load_agentic_cot_coding(
    *,
    max_rows: int | None = None,
    hf_repo: str = "mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    yield from _load_instruction_pairs(
        ds,
        dataset_id="agentic_cot_coding",
        hf_repo=hf_repo,
        max_rows=max_rows,
        verify="agentic_cot",
        user_keys=("user", "instruction", "prompt", "input", "question"),
        assistant_keys=("assistant", "response", "output", "answer", "solution"),
    )


def load_ling_coder_sft(
    *,
    max_rows: int | None = None,
    hf_repo: str = "inclusionAI/Ling-Coder-SFT",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """ShareGPT-style coding SFT (messages + mid)."""
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        messages = _parse_messages(row.get("messages"))
        if not messages or len(messages) < 2:
            continue
        sid = str(row.get("mid") or f"ling-coder-{idx}")
        yield from iter_messages_records(
            dataset_id="ling_coder_sft",
            session_id=sid,
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=True,
            verify="ling_coder_sft",
            extra={"languages": row.get("languages"), "tags": row.get("tags")},
        )
        count += 1


def _nemotron_swe_v2_splits(splits: list[str] | str | None) -> tuple[str, ...]:
    if splits is None:
        return ("agentless",)
    if isinstance(splits, str):
        return tuple(s.strip() for s in splits.split(",") if s.strip())
    return tuple(str(s).strip() for s in splits if str(s).strip())


def load_nemotron_swe_v2(
    *,
    max_rows: int | None = None,
    hf_repo: str = "nvidia/Nemotron-SFT-SWE-v2",
    splits: list[str] | str | None = None,
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """NVIDIA SWE SFT v2 — default split agentless (openhands_swe fails HF streaming cast)."""
    token = _hf_token()
    count = 0
    for split in _nemotron_swe_v2_splits(splits):
        if max_rows is not None and count >= max_rows:
            break
        if split == "openhands_swe":
            print(
                "nemotron_swe_v2: skipping openhands_swe (HF streaming schema cast error); "
                "use agentless only",
                flush=True,
            )
            continue
        try:
            ds = _stream_dataset(hf_repo, split=split, token=token)
        except Exception as exc:
            print(f"nemotron_swe_v2: skip split {split!r} — {exc}", flush=True)
            continue
        for idx, row in enumerate(ds):
            if max_rows is not None and count >= max_rows:
                break
            messages = _parse_messages(row.get("messages"))
            if not messages or len(messages) < 2:
                continue
            sid = str(row.get("uuid") or f"nemotron-swe-v2-{split}-{idx}")
            yield from iter_messages_records(
                dataset_id="nemotron_swe_v2",
                session_id=sid,
                messages=messages,
                hf_repo=hf_repo,
                map_tool_to_user=True,
                verify="nemotron_swe_v2",
                extra={"hf_split": split},
            )
            count += 1


COOPER_QWEN9B_COOP_REPO = "CooperBench/qwen9b-coop-claude-code"
COOPER_TRAJ_FILES = ("agent1_traj.json", "agent2_traj.json")


def _cooper_traj_path(log_dir: str, traj_file: str) -> str:
    log_dir = log_dir.strip().lstrip("/")
    return f"coop/{log_dir}/{traj_file}" if not log_dir.startswith("coop/") else f"{log_dir}/{traj_file}"


def load_cooper_qwen9b_coop_claude(
    *,
    max_rows: int | None = None,
    hf_repo: str = COOPER_QWEN9B_COOP_REPO,
    traj_files: list[str] | str | None = None,
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """
    CooperBench two-agent coop trajectories (Claude Code on Qwen3.5-9B).

    HF index rows point at coop/<repo>/<task>/<features>/agent{1,2}_traj.json.
    """
    if isinstance(traj_files, str):
        files = tuple(s.strip() for s in traj_files.split(",") if s.strip())
    elif traj_files:
        files = tuple(str(f).strip() for f in traj_files if str(f).strip())
    else:
        files = COOPER_TRAJ_FILES

    token = _hf_token()
    fs = None
    root = _active_local_dir(None)
    if root is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=token)
    ds = _stream_dataset(hf_repo, token=token)
    pairs = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and pairs >= max_rows:
            break
        log_dir = str(row.get("log_dir") or "").strip()
        if not log_dir:
            continue
        repo_name = str(row.get("repo") or "repo")
        task_id = row.get("task_id")
        features = str(row.get("features") or "")
        both_passed = row.get("both_passed")
        base_extra = {
            "cooper_setting": row.get("setting"),
            "cooper_model": row.get("model"),
            "both_passed": both_passed,
            "pair_tokens": row.get("pair_tokens"),
            "task_repo": repo_name,
            "task_id": task_id,
            "features": features,
        }
        pair_key = f"{repo_name}-{task_id}-{features}"
        emitted_pair = False
        for traj_file in files:
            rel = _cooper_traj_path(log_dir, traj_file)
            if root is not None:
                local_path = root / rel
                if not local_path.is_file():
                    print(f"cooper_qwen9b_coop: skip {local_path} — missing", flush=True)
                    continue
                try:
                    with local_path.open(encoding="utf-8") as fh:
                        traj = json.load(fh)
                except OSError as exc:
                    print(f"cooper_qwen9b_coop: skip {local_path} — {exc}", flush=True)
                    continue
            else:
                hf_path = f"datasets/{hf_repo}/{rel}"
                try:
                    with fs.open(hf_path, "r") as fh:
                        traj = json.load(fh)
                except OSError as exc:
                    print(f"cooper_qwen9b_coop: skip {hf_path} — {exc}", flush=True)
                    continue
            messages = traj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            agent_id = str(traj.get("agent_id") or traj_file.replace("_traj.json", ""))
            sid = f"{pair_key}-{agent_id}"
            traj_src = str(local_path if root is not None else hf_path)
            yield from iter_messages_records(
                dataset_id="cooper_qwen9b_coop_claude",
                session_id=sid,
                messages=messages,
                hf_repo=hf_repo,
                map_tool_to_user=True,
                verify="cooperbench_coop",
                extra={
                    **base_extra,
                    "agent_id": agent_id,
                    "agent_status": traj.get("status"),
                    "hf_traj_path": traj_src,
                },
            )
            emitted_pair = True
        if emitted_pair:
            pairs += 1


def load_scale_swe(
    *,
    max_rows: int | None = None,
    hf_repo: str = "AweAI-Team/Scale-SWE",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """SWE-bench-style problem_statement + patch pairs."""
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        problem = (row.get("problem_statement") or "").strip()
        patch = (row.get("patch") or "").strip()
        if not problem or not patch:
            continue
        sid = str(row.get("instance_id") or f"scale-swe-{idx}")
        extra = {
            "repo": row.get("repo"),
            "language": row.get("language"),
            "github_url": row.get("github_url"),
        }
        yield make_record(
            dataset_id="scale_swe",
            session_id=sid,
            line_no=1,
            role="user",
            text=problem,
            hf_repo=hf_repo,
            verify="swe_patch",
            extra=extra,
        )
        yield make_record(
            dataset_id="scale_swe",
            session_id=sid,
            line_no=2,
            role="assistant",
            text=patch,
            hf_repo=hf_repo,
            verify="swe_patch",
            extra=extra,
        )
        count += 1


NEMOTRON_OPENCODE_SPLITS = (
    "general",
    "bash_only_tool",
    "bash_only_tool_skills",
    "question_tool",
    "agent_skills",
    "agent_skills_question_tool",
)


def load_nemotron_opencode(
    *,
    max_rows: int | None = None,
    hf_repo: str = "nvidia/Nemotron-SFT-OpenCode-v1",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """OpenCode-style agentic SFT across all HF config splits."""
    count = 0
    token = _hf_token()
    for split in NEMOTRON_OPENCODE_SPLITS:
        if max_rows is not None and count >= max_rows:
            break
        ds = _stream_hf_jsonl(hf_repo, f"{split}/data.jsonl", token=token)
        for idx, row in enumerate(ds):
            if max_rows is not None and count >= max_rows:
                break
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            has_tool = any(
                isinstance(m, dict) and (m.get("role") or "").lower() == "tool"
                for m in messages
            )
            sid = str(row.get("uuid") or f"nemotron-opencode-{split}-{idx}")
            yield from iter_messages_records(
                dataset_id="nemotron_opencode",
                session_id=sid,
                messages=messages,
                hf_repo=hf_repo,
                map_tool_to_user=has_tool,
                verify="nemotron_opencode",
                extra={
                    "hf_split": split,
                    "question_category": row.get("question_category"),
                    "complexity_level": row.get("complexity_level"),
                    "enabled_tools": row.get("enabled_tools"),
                },
            )
            count += 1


def load_coderforge_preview(
    *,
    max_rows: int | None = None,
    hf_repo: str = "togethercomputer/CoderForge-Preview",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """Long-horizon agent trajectories; default split is reward-filtered successes."""
    ds = _stream_dataset(hf_repo, config="trajectories", split="filtered_reward1")
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        messages = _parse_messages_field(row.get("messages"))
        if len(messages) < 2:
            continue
        sid = str(row.get("trajectory_id") or f"coderforge-{idx}")
        yield from iter_messages_records(
            dataset_id="coderforge_preview",
            session_id=sid,
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=True,
            verify="coderforge_trajectory",
            extra={
                "reward": row.get("reward"),
                "finish_reason": row.get("finish_reason"),
                "license": row.get("license"),
            },
        )
        count += 1
