"""Doc-aligned preflight checks before train-qlora (Unsloth or Chronicals)."""

from __future__ import annotations

import importlib.util

from llm_train.quiet import apply_train_quiet, suppress_unsloth_import_noise

apply_train_quiet()

if importlib.util.find_spec("unsloth") is not None:
    with suppress_unsloth_import_noise():
        import unsloth  # noqa: F401, E402

import json
import sys
from pathlib import Path

from llm_train.config import (
    chronicals_settings,
    default_train_backend,
    default_train_file,
    train_settings,
    unsloth_settings,
)
from llm_train.dataset import load_messages_dataset, train_file_stats
from llm_train.dataset_filter import filter_dataset_to_max_tokens, max_chars_for_seq
from llm_train.chronicals_runtime import (
    _flash_attn_available,
    load_train_tokenizer,
)
from llm_train.flash_attn import flash_attn_available
from llm_train.token_audit import audit_messages_lengths, print_token_audit
from llm_train.vram_budget import (
    plan_unsloth_training,
    resolve_vram_train_params,
    unsloth_vram_seq_ceiling,
)


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def run_preflight(
    *,
    promote: bool = False,
    decensor: bool = False,
    train_file: Path | None = None,
    backend: str | None = None,
) -> int:
    """Return 0 if all hard checks pass."""
    backend = (backend or default_train_backend()).lower()
    cfg = train_settings(promote=promote, decensor=decensor)
    unsloth = unsloth_settings(promote=promote, decensor=decensor)
    chronicals = chronicals_settings(promote=promote, decensor=decensor)
    path = train_file or default_train_file()
    errors = 0

    print("=== train-preflight ===")
    profile = "decensor" if decensor else ("promote" if promote else "bootstrap")
    print(f"profile: {profile}")
    print(f"backend: {backend}")
    if decensor:
        from llm_train.config import decensor_settings

        dec = decensor_settings()
        print(f"base_model: {dec['base_model']}")
    print(f"train_file: {path}")
    print()

    # --- deps ---
    print("[deps]")
    for mod in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"):
        try:
            __import__(mod)
            _ok(mod)
        except ImportError:
            _fail(f"missing package: {mod} (run: uv sync --package llm-train --extra unsloth)")
            errors += 1

    if backend == "unsloth":
        try:
            import importlib.metadata
            import importlib.util

            if importlib.util.find_spec("unsloth") is None:
                _fail("missing unsloth (run: uv sync --package llm-train --extra unsloth)")
                errors += 1
            else:
                import unsloth as _unsloth_pkg  # noqa: F401

                _ok("unsloth")
                try:
                    importlib.metadata.version("unsloth_zoo")
                    _ok("unsloth_zoo")
                except importlib.metadata.PackageNotFoundError:
                    _fail(
                        "missing unsloth_zoo — run: uv pip install 'unsloth-zoo>=2026.5.4' --no-deps "
                        "&& uv pip install pillow sentencepiece protobuf hf-transfer cut-cross-entropy torchao"
                    )
                    errors += 1
        except ImportError as exc:
            _fail(f"unsloth import failed: {exc}")
            errors += 1
    else:
        try:
            __import__("chronicals")
            _ok("chronicals")
        except ImportError:
            _fail("missing chronicals (run: uv sync --package llm-train --extra chronicals)")
            errors += 1

    import torch

    from llm_core.gpu_mutex import gpu_ghost_entries, gpu_vram_recovery_instructions

    if not torch.cuda.is_available():
        _fail("CUDA not available")
        errors += 1
    else:
        free, total = torch.cuda.mem_get_info()
        _ok(f"GPU {torch.cuda.get_device_name(0)} — {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total")
        if free / 2**30 < 8.0:
            _warn("under 8 GiB free — stop hyprwhspr/Ollama before train")
        ghosts = gpu_ghost_entries()
        ghost_mib = sum(m for _, _, m in ghosts)
        if ghost_mib >= 1000:
            _fail(
                f"ghost VRAM ~{ghost_mib} MiB from dead GPU process(es) — "
                "log out/in or reboot before train"
            )
            for pid, name, mem in ghosts:
                print(f"         pid={pid} {name} ({mem} MiB)", file=sys.stderr)
            print(gpu_vram_recovery_instructions(), file=sys.stderr)
            errors += 1

    print()

    # --- TRL chat template ---
    print("[TRL chat template]")
    if not path.is_file():
        _fail(f"missing train file: {path}")
        return errors + 1

    stats = train_file_stats(path)
    print(json.dumps({"train_file_stats": stats}, indent=2))

    model_id = cfg["base_model"]
    if backend == "unsloth" and not decensor:
        from llm_train.unsloth_runtime import resolve_unsloth_model_id

        model_id = resolve_unsloth_model_id(cfg, unsloth)

    try:
        from trl.chat_template_utils import get_training_chat_template

        if backend == "chronicals":
            tok = load_train_tokenizer(model_id, max_seq=int(cfg["max_seq_length"]))
        else:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tpl = get_training_chat_template(tok)
        if tpl:
            _ok("get_training_chat_template → patched qwen2_5_training template")
        else:
            _ok("native template already has {% generation %} markers")
    except Exception as exc:
        _fail(f"get_training_chat_template: {exc}")
        errors += 1

    print()

    # --- dataset ---
    print("[dataset / assistant_only_loss]")
    import torch

    if torch.cuda.is_available():
        from llm_core.gpu_mutex import reclaim_gpu_before_load

        if reclaim_gpu_before_load():
            _ok("GPU reclaimed for VRAM plan (same as train-qlora — close browser if seq stuck @ 768)")
        else:
            _warn(
                "GPU reclaim incomplete (<8 GiB free) — close browser/Ollama; "
                "2048 needs flash-attn + ~10 GiB free"
            )
    free_gb, total_gb = (x / (1024**3) for x in torch.cuda.mem_get_info()) if torch.cuda.is_available() else (0.0, 0.0)
    if torch.cuda.is_available():
        _ok(f"VRAM after reclaim: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")

    token_rec: int | None = None
    vram_ceiling: int | None = None
    if backend == "unsloth":
        vram_ceiling, ceiling_note = unsloth_vram_seq_ceiling(
            cfg, unsloth, free_gb=free_gb, total_gb=total_gb
        )
        _ok(ceiling_note)

    if backend == "unsloth" and unsloth.get("token_audit", True):
        print("[token audit]")
        try:
            from llm_train.unsloth_runtime import load_unsloth_tokenizer

            audit_tok = load_unsloth_tokenizer(cfg, unsloth)
            char_cap_pre = int(cfg["max_chars_per_message"]) if cfg.get("max_chars_per_message") else max_chars_for_seq(
                min(int(cfg["max_seq_length"]), vram_ceiling or int(cfg["max_seq_length"]))
            )
            audit_ds, _ = load_messages_dataset(path, max_chars_per_message=char_cap_pre)
            report, _ = audit_messages_lengths(
                audit_ds,
                audit_tok,
                yaml_cap=int(cfg["max_seq_length_12gb_cap"]),
                vram_ceiling=int(vram_ceiling or cfg["max_seq_length_12gb_cap"]),
            percentile=float(unsloth.get("token_audit_percentile", 95)),
            round_to=int(unsloth.get("token_audit_round_to", 256)),
            min_seq=int(unsloth.get("token_audit_min_seq", 512)),
            headroom_ratio=float(unsloth.get("token_audit_headroom_ratio", 1.05)),
            )
            print_token_audit(report)
            token_rec = report.recommended_seq
            if report.effective_cap < int(cfg["max_seq_length"]):
                _ok(
                    f"effective cap {report.effective_cap} < yaml aspire {cfg['max_seq_length']} "
                    "(VRAM-bound — install flash-attn to unlock 2048)"
                )
            if report.would_drop_assistant_at_cap > len(audit_ds) * 0.25:
                _warn(
                    f"~{report.would_drop_assistant_at_cap} rows may lose assistant @ "
                    f"cap {report.effective_cap} — chunk long chats in dataprep"
                )
        except Exception as exc:
            _warn(f"token audit skipped: {exc}")
        print()

    if backend == "unsloth":
        plan = plan_unsloth_training(
            cfg,
            unsloth,
            smoke=False,
            free_gb=free_gb,
            total_gb=total_gb,
            token_recommended_seq=token_rec,
        )
        max_seq, batch, grad_accum = plan.max_seq, plan.batch_size, plan.grad_accum
        vram_reason = plan.reason
    else:
        max_seq, batch, grad_accum, vram_reason = resolve_vram_train_params(
            cfg,
            smoke=False,
            free_gb=free_gb,
            total_gb=total_gb,
            chronicals=chronicals,
            backend=backend,
            unsloth=None,
            token_recommended_seq=token_rec,
        )
    _ok(f"VRAM plan: seq<={max_seq} batch={batch} grad_accum={grad_accum} ({vram_reason})")

    char_cap = int(cfg["max_chars_per_message"]) if cfg.get("max_chars_per_message") else max_chars_for_seq(max_seq)
    raw_ds, weights = load_messages_dataset(path, max_chars_per_message=char_cap)
    _ok(f"loaded {len(raw_ds)} rows (char_cap={char_cap})")

    if backend == "unsloth":
        _ok("Unsloth pre-tokenizes messages (TRL template + assistant_masks; unsloth-zoo#323)")
        kept = len(raw_ds)
        dropped = 0
    else:
        tok = load_train_tokenizer(model_id, max_seq=max_seq)
        print("Running TRL assistant-only filter…", flush=True)
        filtered, weights = filter_dataset_to_max_tokens(
            raw_ds, tok, max_seq=max_seq, sample_weights=weights
        )
        kept, dropped = len(filtered), len(raw_ds) - len(filtered)
        if kept == 0:
            _fail("zero rows after assistant-only filter — cannot train")
            errors += 1
        else:
            _ok(f"{kept} rows kept, {dropped} dropped (no assistant tokens @ keep_end truncate)")
            if dropped > len(raw_ds) * 0.2:
                _warn(f"{100 * dropped / len(raw_ds):.0f}% dropped — consider lower max_seq")

    print()

    if backend == "unsloth":
        print("[Unsloth runtime]")
        _ok(f"model: {model_id}")
        fa = flash_attn_available()
        if fa:
            _ok("flash-attn available (2048 + padding-free or BFD pack)")
        else:
            _warn(
                "flash-attn not installed — seq capped, no padding-free/BFD pack. "
                "Run: bash scripts/install-flash-attn.sh (needs ~10 GiB free VRAM at train time)"
            )
        _ok(f"gradient_checkpointing={unsloth.get('use_gradient_checkpointing', 'unsloth')}")
        _ok(f"RSLoRA={unsloth.get('use_rslora', True)}")
        _ok(f"max_grad_norm={unsloth.get('max_grad_norm', 1.0)}")
        _ok(f"effective_batch_target={unsloth.get('effective_batch_target', 16)}")
        if unsloth.get("clamp_to_one_epoch", True):
            _ok("clamp_to_one_epoch (quality — one pass unless --max-steps)")
        if unsloth.get("eval_holdout_ratio", 0) > 0:
            _ok(f"eval holdout {unsloth.get('eval_holdout_ratio', 0):.0%} + load_best_model_at_end")
        if unsloth.get("disable_torch_compile", True):
            _ok("UNSLOTH_COMPILE_DISABLE (stable 12GB step-0)")
        from llm_train.unsloth_runtime import resolve_unsloth_runtime_flags

        rt = resolve_unsloth_runtime_flags(unsloth)
        if rt["use_padding_free"]:
            _ok("padding-free ON (FA2 — keeps weighted sampler)")
        elif rt["use_packing"]:
            _ok(f"BFD pack ON (FA2, strategy={unsloth.get('packing_strategy', 'bfd')})")
        elif not fa:
            _ok("auto padding-free disabled (no FA2 / TRL max_length safe path)")
        if unsloth.get("use_lora_plus", True):
            _ok(f"LoRA+ ratio={unsloth.get('lora_plus_lr_ratio', 16)}")
        if unsloth.get("stratified_eval_holdout", True):
            _ok(f"stratified eval holdout {unsloth.get('eval_holdout_ratio', 0.1):.0%}")
        if unsloth.get("activation_offloading") and fa:
            _ok("activation_offloading ON when FA2 (promote)")
    else:
        print("[TRL packing / flash-attn]")
        fa = _flash_attn_available()
        want_packing = bool(chronicals.get("use_sequence_packing"))
        if want_packing and not fa:
            _warn("use_sequence_packing=true but flash-attn not installed — runtime disables packing")
        elif want_packing and fa:
            _ok("flash-attn + packing enabled")
        else:
            _ok("packing off (expected without flash-attn)")

        print()
        print("[Chronicals patches]")
        if chronicals.get("use_activation_offload"):
            _ok("activation offload ON (promote)")
        if chronicals.get("use_chronicals_gradient_checkpointing"):
            _ok("Chronicals sqrt(n) gradient checkpointing ON")
        if chronicals.get("use_lora_plus"):
            _ok(f"LoRA+ ratio={chronicals.get('lora_plus_lr_ratio', 16)}")
        if chronicals.get("use_liger_kernel"):
            _warn("Liger ON — class-level only; fused LCE stays off per TRL+PEFT")

    print()

    # --- config sanity ---
    print("[config]")
    steps_per_epoch = max(1, kept // (batch * grad_accum))
    cap = int(cfg["max_steps_cap"])
    if backend == "unsloth" and unsloth.get("clamp_to_one_epoch", True):
        _ok(f"~{steps_per_epoch} steps/epoch (max_steps defaults to one epoch)")
    else:
        _ok(f"~{steps_per_epoch} steps/epoch, max_steps_cap={cap}")
        if cap > steps_per_epoch:
            _warn(f"cap {cap} > one epoch ({steps_per_epoch}) — multi-pass unless --max-steps overrides")

    print()
    if errors:
        print(f"PREFLIGHT FAILED ({errors} hard error(s))")
        return 1
    print("PREFLIGHT PASSED — safe to run train-qlora")
    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Doc-aligned preflight before train-qlora")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--decensor", action="store_true")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--backend", choices=("unsloth", "chronicals"), default=None)
    parser.add_argument("--chronicals", action="store_true")
    args = parser.parse_args()
    backend = "chronicals" if args.chronicals else args.backend
    raise SystemExit(
        run_preflight(
            promote=args.promote,
            decensor=args.decensor,
            train_file=args.train_file,
            backend=backend,
        )
    )
