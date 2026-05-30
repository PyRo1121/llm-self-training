"""Hugging Face dataset loaders → raw JSONL records."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from llm_dataprep.public.records import iter_messages_records, make_record

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


def _stream_dataset(
    repo: str,
    *,
    split: str = "train",
    data_files: str | dict[str, str] | None = None,
    config: str | None = None,
    token: str | None = None,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
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
) -> Iterator[dict[str, Any]]:
    """Stream JSONL from an HF dataset repo (tolerates mixed column types)."""
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
    by_session: dict[str, list[dict[str, Any]]] = {}
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
        by_session.setdefault(sid, []).append(
            {"role": out_role, "content": prefix + content, "turn": row.get("turn_number", 0)}
        )
        emitted += 1

    for sid, turns in by_session.items():
        turns.sort(key=lambda t: t.get("turn", 0))
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]
        yield from iter_messages_records(
            dataset_id="swe_chat",
            session_id=sid,
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=False,
            verify="swe_chat_wild",
            extra={"public_release": "2026-04"},
        )


def load_opencode_broad(
    *,
    max_rows: int | None = None,
    hf_repo: str = "EER6/nvidia-OpenCodeInstruct-broad",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    return _load_opencode_pair(hf_repo=hf_repo, dataset_id="opencode_broad", max_rows=max_rows)


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


def load_nemotron_swe(
    *,
    max_rows: int | None = None,
    hf_repo: str = "nvidia/Nemotron-Cascade-SFT-SWE",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        sid = f"nemotron-swe-{idx}"
        yield from iter_messages_records(
            dataset_id="nemotron_swe",
            session_id=sid,
            messages=messages,
            hf_repo=hf_repo,
            map_tool_to_user=False,
            verify="nemotron_swe",
            extra={"category": row.get("category"), "source": row.get("source")},
        )
        count += 1


def load_self_code_align(
    *,
    max_rows: int | None = None,
    hf_repo: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        user = (row.get("instruction") or row.get("prompt") or "").strip()
        assistant = (row.get("response") or "").strip()
        if not user or not assistant:
            continue
        sid = str(row.get("id") or row.get("fingerprint") or f"sc2-{idx}")
        yield make_record(
            dataset_id="self_code_align",
            session_id=sid,
            line_no=1,
            role="user",
            text=user,
            hf_repo=hf_repo,
            verify="exec_filtered",
        )
        yield make_record(
            dataset_id="self_code_align",
            session_id=sid,
            line_no=2,
            role="assistant",
            text=assistant,
            hf_repo=hf_repo,
            verify="exec_filtered",
        )
        count += 1


def _load_instruction_pair(
    *,
    hf_repo: str,
    dataset_id: str,
    max_rows: int | None,
    verify: str,
    user_keys: tuple[str, ...] = ("input", "instruction", "problem", "prompt"),
    assistant_keys: tuple[str, ...] = ("output", "response", "solution"),
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        user = ""
        for key in user_keys:
            user = (row.get(key) or "").strip()
            if user:
                break
        assistant = ""
        for key in assistant_keys:
            assistant = (row.get(key) or "").strip()
            if assistant:
                break
        if not user or not assistant:
            continue
        sid = str(row.get("id") or row.get("index") or f"{dataset_id}-{idx}")
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=1,
            role="user",
            text=user,
            hf_repo=hf_repo,
            verify=verify,
        )
        yield make_record(
            dataset_id=dataset_id,
            session_id=sid,
            line_no=2,
            role="assistant",
            text=assistant,
            hf_repo=hf_repo,
            verify=verify,
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


ULTRADATA_CONFIGS: tuple[str, ...] = ("Knowledge", "Code", "Math")
ULTRADATA_SPLIT = "no_think"  # also available: think (reasoning traces)


def load_ultradata_sft_2605(
    *,
    max_rows: int | None = None,
    hf_repo: str = "openbmb/UltraData-SFT-2605",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    """UltraData multi-config SFT — Knowledge, Code, Math (no_think split)."""
    token = _hf_token()
    count = 0
    for cfg in ULTRADATA_CONFIGS:
        if max_rows is not None and count >= max_rows:
            break
        cap = None if max_rows is None else max_rows - count
        ds = _stream_dataset(hf_repo, config=cfg, split=ULTRADATA_SPLIT, token=token)
        for idx, row in enumerate(ds):
            if cap is not None and count >= cap:
                break
            messages = _parse_messages(row.get("messages"))
            if not messages:
                continue
            sid = str(row.get("uid") or f"ultradata-{cfg}-{idx}")
            extra = {
                "ultradata_config": cfg,
                "ultradata_split": ULTRADATA_SPLIT,
                "domain": row.get("domain"),
                "source_subset": row.get("source"),
            }
            yield from iter_messages_records(
                dataset_id="ultradata_sft_2605",
                session_id=sid,
                messages=messages,
                hf_repo=hf_repo,
                map_tool_to_user=False,
                verify="ultradata_sft",
                extra=extra,
            )
            count += 1


def load_high_coder_sft(
    *,
    max_rows: int | None = None,
    hf_repo: str = "Crownelius/High-Coder-SFT-Medium",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        user = (provenance.get("prompt") or "").strip()
        content = row.get("content") if isinstance(row.get("content"), dict) else {}
        assistant = (content.get("text") or "").strip()
        if not user or not assistant:
            continue
        sid = str(row.get("sample_id") or f"high-coder-{idx}")
        lang = row.get("language")
        loc = None
        lc = row.get("long_criteria")
        if isinstance(lc, dict):
            loc = lc.get("loc")
        extra = {"language": lang, "loc": loc}
        yield make_record(
            dataset_id="high_coder_sft",
            session_id=sid,
            line_no=1,
            role="user",
            text=user,
            hf_repo=hf_repo,
            verify="synthetic_longform",
            extra=extra,
        )
        yield make_record(
            dataset_id="high_coder_sft",
            session_id=sid,
            line_no=2,
            role="assistant",
            text=assistant,
            hf_repo=hf_repo,
            verify="synthetic_longform",
            extra=extra,
        )
        count += 1


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


def load_agent_trove(
    *,
    max_rows: int | None = None,
    hf_repo: str = "open-thoughts/AgentTrove",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    yield from _load_instruction_pairs(
        ds,
        dataset_id="agent_trove",
        hf_repo=hf_repo,
        max_rows=max_rows,
        verify="agent_trove",
    )


def load_codex_7m(
    *,
    max_rows: int | None = None,
    hf_repo: str = "Modotte/CodeX-7M-Non-Thinking",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    return _load_instruction_pair(
        hf_repo=hf_repo,
        dataset_id="codex_7m",
        max_rows=max_rows,
        verify="codex_instruction",
    )


def load_codex_2m_thinking(
    *,
    max_rows: int | None = None,
    hf_repo: str = "Modotte/CodeX-2M-Thinking",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    return _load_instruction_pair(
        hf_repo=hf_repo,
        dataset_id="codex_2m_thinking",
        max_rows=max_rows,
        verify="codex_thinking",
    )


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


def load_magicoder(
    *,
    max_rows: int | None = None,
    hf_repo: str = "ise-uiuc/Magicoder-OSS-Instruct-75K",
    **_: Any,
) -> Iterator[dict[str, Any]]:
    ds = _stream_dataset(hf_repo)
    count = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and count >= max_rows:
            break
        user = (row.get("problem") or row.get("instruction") or "").strip()
        assistant = (row.get("solution") or row.get("response") or "").strip()
        if not user or not assistant:
            continue
        sid = str(row.get("index") or row.get("raw_index") or f"magicoder-{idx}")
        yield make_record(
            dataset_id="magicoder_75k",
            session_id=sid,
            line_no=1,
            role="user",
            text=user,
            hf_repo=hf_repo,
            verify="synthetic_oss",
        )
        yield make_record(
            dataset_id="magicoder_75k",
            session_id=sid,
            line_no=2,
            role="assistant",
            text=assistant,
            hf_repo=hf_repo,
            verify="synthetic_oss",
        )
        count += 1
