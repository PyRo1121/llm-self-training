"""Phase 2 — QLoRA SFT (4070 Ti, Qwen2.5-Coder-7B). Default backend: Unsloth."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from llm_core.control_plane import register_training_run
from llm_core.gpu_mutex import GpuMutex, load_gpu_mutex_settings
from llm_train.config import (
    chronicals_settings,
    default_output_dir,
    default_train_backend,
    default_train_file,
    train_settings,
    unsloth_settings,
)
from llm_core.paths import config_dir
from llm_train.dataset import (
    load_messages_dataset,
    sample_weights_from_dataset,
    train_file_stats,
)
from llm_train.dataset_filter import filter_dataset_to_max_tokens, max_chars_for_seq
from llm_train.vram_budget import (
    _hard_max_seq,
    log_vram_snapshot,
    resolve_vram_train_params,
)


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame) -> None:
        names = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}
        label = names.get(signum, f"signal {signum}")
        print(f"\nTrain interrupted: {label}", file=sys.stderr, flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA not available. Training requires a GPU (4070 Ti). "
            "Use --dry-run to validate the dataset only.",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_backend(args) -> str:
    if getattr(args, "chronicals", False):
        return "chronicals"
    if args.backend:
        return args.backend.lower()
    return default_train_backend()


def _attach_weighted_sampler(trainer, cfg: dict) -> None:
    import torch
    from torch.utils.data import WeightedRandomSampler

    n_train = len(trainer.train_dataset)
    weight_tensor = torch.tensor(
        sample_weights_from_dataset(trainer.train_dataset, cfg),
        dtype=torch.double,
    )
    if len(weight_tensor) != n_train:
        print(
            f"Warning: sample_weights ({len(weight_tensor)}) != train rows ({n_train}); "
            "using uniform sampler.",
            file=sys.stderr,
        )
        weight_tensor = torch.ones(n_train, dtype=torch.double)

    def _weighted_sampler(_dataset):
        return WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=n_train,
            replacement=True,
        )

    trainer._get_train_sampler = _weighted_sampler


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT — Unsloth (default) or Chronicals")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="max_steps=5, cap 128 examples")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Quality profile (seq 2048, r=32, LR 2e-4, 400 steps)",
    )
    parser.add_argument(
        "--decensor",
        action="store_true",
        help="Promote profile on abliterated base",
    )
    parser.add_argument(
        "--backend",
        choices=("unsloth", "chronicals"),
        default=None,
        help="Train stack (default: train.backend in config, usually unsloth)",
    )
    parser.add_argument(
        "--chronicals",
        action="store_true",
        help="Use Chronicals+TRL backend instead of Unsloth",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dataset stats only, no GPU")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--no-gpu-mutex", action="store_true")
    parser.add_argument("--gpu-reclaim-warn-only", action="store_true")
    parser.add_argument("--gpu-reclaim-conservative", action="store_true")
    args = parser.parse_args()

    backend = _resolve_backend(args)
    cfg = train_settings(promote=args.promote, decensor=args.decensor)
    unsloth = unsloth_settings(promote=args.promote, decensor=args.decensor)
    chronicals = chronicals_settings(promote=args.promote, decensor=args.decensor)
    train_path = args.train_file or default_train_file()

    if backend == "unsloth":
        import importlib.util

        if importlib.util.find_spec("unsloth") is None:
            print(
                "Unsloth not installed. Run:\n"
                "  uv sync --package llm-train --extra unsloth\n"
                "Or use --chronicals for the legacy backend.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not train_path.is_file():
        print(
            f"Missing {train_path}. Run:\n"
            "  uv run --package llm-dataprep training-manifest --manifest-id personal-first\n"
            "  uv run --package llm-dataprep training-extract "
            f"--manifest-id personal-first --out {train_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.decensor:
        from llm_train.config import decensor_settings

        dec = decensor_settings()
        print(
            f"Decensor profile ({backend}): base={dec['base_model']} — "
            "see docs/CODING-SAFEGUARDS.md",
            flush=True,
        )
    elif args.promote:
        if backend == "unsloth":
            print(
                "Promote profile (Unsloth): seq≤2048, r=32, all-linear, RSLoRA, "
                "LR=1.5e-4, eff_batch≈12, max_grad_norm=1.0",
                flush=True,
            )
        else:
            print(
                "Promote profile (Chronicals): seq≤1024, LoRA r=32, activation offload",
                flush=True,
            )
    else:
        print(f"Backend: {backend}", flush=True)

    stats = train_file_stats(train_path)
    print(json.dumps({"train_file_stats": stats}, indent=2))

    if args.dry_run:
        return

    _require_cuda()
    _install_signal_handlers()

    if backend == "unsloth":
        from llm_train.unsloth_runtime import apply_unsloth_env

        apply_unsloth_env(unsloth)
    else:
        from llm_train.chronicals_runtime import apply_chronicals_env

        apply_chronicals_env(chronicals)

    gpu_cfg = load_gpu_mutex_settings()
    with GpuMutex(
        settings=gpu_cfg,
        enabled=not args.no_gpu_mutex,
        stop_hyprwhspr_service=not args.no_gpu_mutex,
        stop_ollama_models=not args.no_gpu_mutex,
        restore_hyprwhspr=not args.no_gpu_mutex,
        warn_only=args.gpu_reclaim_warn_only,
        reclaim_unknown=False if args.gpu_reclaim_conservative else None,
    ):
        log_vram_snapshot()
        if backend == "unsloth":
            _run_train_unsloth(args, cfg, unsloth, train_path, stats)
        else:
            _run_train_chronicals(args, cfg, chronicals, train_path, stats)


def _run_train_unsloth(args, cfg: dict, unsloth: dict, train_path: Path, stats: dict) -> None:
    from trl import SFTTrainer

    from llm_train.unsloth_runtime import (
        apply_train_on_responses_only,
        build_unsloth_sft_config,
        load_unsloth_model,
    )

    out_dir = args.output_dir or default_output_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "backend": "unsloth",
                "settings": cfg,
                "unsloth": unsloth,
                "train_file_stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import torch

    free_gb, total_gb = (x / (1024**3) for x in torch.cuda.mem_get_info())
    max_seq, batch_size, grad_accum, vram_reason = resolve_vram_train_params(
        cfg,
        smoke=args.smoke,
        free_gb=free_gb,
        total_gb=total_gb,
        backend="unsloth",
        unsloth=unsloth,
    )
    print(
        f"VRAM budget: {vram_reason} → batch={batch_size} "
        f"seq<={max_seq} grad_accum={grad_accum}",
        file=sys.stderr,
        flush=True,
    )
    torch.cuda.empty_cache()

    max_examples = 128 if args.smoke else None
    raw_chars = cfg.get("max_chars_per_message")
    char_cap = int(raw_chars) if raw_chars is not None else max_chars_for_seq(max_seq)
    dataset, _sample_weights = load_messages_dataset(
        train_path,
        max_examples=max_examples,
        max_chars_per_message=char_cap,
    )
    print(f"Dataset char cap per message: {char_cap}", flush=True)
    print(f"Training examples (Unsloth will filter empty assistant @ truncate): {len(dataset)}")

    if args.lora_r:
        cfg = {**cfg, "lora_r": args.lora_r}

    model, tokenizer = load_unsloth_model(cfg, unsloth, max_seq)

    epochs = args.epochs if args.epochs is not None else cfg["num_epochs"]
    n = len(dataset)
    steps_per_epoch = max(1, n // (batch_size * grad_accum))
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = min(int(steps_per_epoch * epochs), cfg["max_steps_cap"])
    if args.smoke:
        max_steps = 5

    eff_batch = batch_size * grad_accum
    steps_note = (
        f"max_steps={max_steps} (effective_batch={eff_batch}, "
        f"~{steps_per_epoch} steps/epoch, cap={cfg['max_steps_cap']})"
    )

    sft_config = build_unsloth_sft_config(
        cfg=cfg,
        unsloth=unsloth,
        out_dir=str(out_dir),
        max_seq=max_seq,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        warmup_ratio=cfg["warmup_ratio"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer = apply_train_on_responses_only(trainer, unsloth)
    _attach_weighted_sampler(trainer, cfg)

    run_name = out_dir.name
    n_train = len(trainer.train_dataset)
    register_training_run(
        run_name,
        base_model=cfg["base_model"],
        status="running",
        train_rows=n_train,
    )

    print(f"Training {n_train} examples, {steps_note}, output={out_dir}")
    torch.cuda.empty_cache()
    try:
        trainer.train()
    except Exception:
        register_training_run(
            run_name,
            base_model=cfg["base_model"],
            status="failed",
            train_rows=n_train,
        )
        raise

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved → {adapter_dir}")

    register_training_run(
        run_name,
        base_model=cfg["base_model"],
        adapter_path=str(adapter_dir),
        status="completed",
        train_rows=n_train,
        metrics={
            "backend": "unsloth",
            "max_steps": max_steps,
            "max_seq": max_seq,
            "effective_batch": eff_batch,
        },
    )


def _run_train_chronicals(
    args, cfg: dict, chronicals: dict, train_path: Path, stats: dict
) -> None:
    from trl import SFTTrainer

    from llm_train.chronicals_runtime import (
        apply_chronicals_env,
        build_sft_config,
        create_lora_plus_optimizer,
        ensure_transformers_config,
        load_qlora_model,
        load_train_tokenizer,
    )

    apply_chronicals_env(chronicals)

    out_dir = args.output_dir or default_output_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "backend": "chronicals",
                "settings": cfg,
                "chronicals": chronicals,
                "train_file_stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import torch

    free_gb, total_gb = (x / (1024**3) for x in torch.cuda.mem_get_info())
    max_seq, batch_size, grad_accum, vram_reason = resolve_vram_train_params(
        cfg,
        smoke=args.smoke,
        free_gb=free_gb,
        total_gb=total_gb,
        chronicals=chronicals,
        backend="chronicals",
    )
    print(
        f"VRAM budget: {vram_reason} → batch={batch_size} "
        f"seq<={max_seq} grad_accum={grad_accum} "
        f"(12GB seq ceiling={_hard_max_seq(chronicals)}, config={config_dir() / 'default.yaml'})",
        file=sys.stderr,
        flush=True,
    )
    torch.cuda.empty_cache()

    max_examples = 128 if args.smoke else None
    raw_chars = cfg.get("max_chars_per_message")
    char_cap = int(raw_chars) if raw_chars is not None else max_chars_for_seq(max_seq)
    dataset, sample_weights = load_messages_dataset(
        train_path,
        max_examples=max_examples,
        max_chars_per_message=char_cap,
    )
    print(f"Dataset char cap per message: {char_cap}", flush=True)

    if args.lora_r:
        cfg = {**cfg, "lora_r": args.lora_r}

    print("Preparing dataset (tokenizer filter, CPU-only)…", flush=True)
    tokenizer = load_train_tokenizer(cfg["base_model"], max_seq=max_seq)
    dataset, sample_weights = filter_dataset_to_max_tokens(
        dataset, tokenizer, max_seq=max_seq, sample_weights=sample_weights
    )
    print(f"Training examples after filter: {len(dataset)}", flush=True)

    model, tokenizer, peft_cfg, liger_applied = load_qlora_model(
        cfg, chronicals, max_seq, tokenizer=tokenizer
    )
    ensure_transformers_config(model, cfg["base_model"], chronicals)

    epochs = args.epochs if args.epochs is not None else cfg["num_epochs"]
    n = len(dataset)
    steps_per_epoch = max(1, n // (batch_size * grad_accum))
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = min(int(steps_per_epoch * epochs), cfg["max_steps_cap"])
    if args.smoke:
        max_steps = 5

    eff_batch = batch_size * grad_accum
    steps_note = (
        f"max_steps={max_steps} (effective_batch={eff_batch}, "
        f"~{steps_per_epoch} steps/epoch, cap={cfg['max_steps_cap']})"
    )

    sft_config = build_sft_config(
        cfg=cfg,
        chronicals=chronicals,
        out_dir=str(out_dir),
        max_seq=max_seq,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        warmup_ratio=cfg["warmup_ratio"],
        liger_applied=liger_applied,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_cfg,
        args=sft_config,
    )
    ensure_transformers_config(trainer.model, cfg["base_model"], chronicals)

    train_model = trainer.model
    if chronicals.get("use_gradient_checkpointing", True) and hasattr(
        train_model, "gradient_checkpointing_enable"
    ):
        train_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_plus = create_lora_plus_optimizer(train_model, cfg, chronicals)
    if lora_plus is not None:
        trainer.optimizer = lora_plus

    _attach_weighted_sampler(trainer, cfg)

    run_name = out_dir.name
    n_train = len(trainer.train_dataset)
    register_training_run(
        run_name,
        base_model=cfg["base_model"],
        status="running",
        train_rows=n_train,
    )

    print(f"Training {n_train} examples, {steps_note}, output={out_dir}")
    torch.cuda.empty_cache()
    try:
        trainer.train()
    except Exception:
        register_training_run(
            run_name,
            base_model=cfg["base_model"],
            status="failed",
            train_rows=n_train,
        )
        raise

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved → {adapter_dir}")

    register_training_run(
        run_name,
        base_model=cfg["base_model"],
        adapter_path=str(adapter_dir),
        status="completed",
        train_rows=n_train,
        metrics={
            "backend": "chronicals",
            "max_steps": max_steps,
            "max_seq": max_seq,
            "effective_batch": eff_batch,
            "liger_applied": liger_applied,
        },
    )


if __name__ == "__main__":
    main()
