"""Merge QLoRA adapter and export for Ollama (Unsloth GGUF or HF merge + llama.cpp)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from llm_train.config import exports_dir, train_settings, unsloth_settings


def _export_unsloth_gguf(adapter_dir: Path, out: Path, quant: str) -> None:
    from unsloth import FastLanguageModel

    from llm_train.unsloth_runtime import ensure_unsloth_imported, resolve_unsloth_model_id

    ensure_unsloth_imported()
    cfg = train_settings()
    unsloth = unsloth_settings()
    model_id = resolve_unsloth_model_id(cfg, unsloth)

    print(f"Loading base {model_id} + adapter {adapter_dir} for GGUF export…", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model.load_adapter(str(adapter_dir))
    out.mkdir(parents=True, exist_ok=True)
    gguf_dir = out / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic 2.0 when available; else q4_k_m (Unsloth save.py)
    methods = [quant]
    if quant == "q4_k_m":
        methods = ["q4_k_m"]

    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=methods)
    print(f"Unsloth GGUF export → {gguf_dir}")
    print(f"Next: ollama create pyro-coder:7b -f {out}/Modelfile")


def _export_hf_merge(adapter_dir: Path, out: Path, quant: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        print("CUDA required for merge export", file=sys.stderr)
        sys.exit(1)

    cfg = train_settings()
    merged = out / "merged-hf"
    merged.mkdir(parents=True, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer.save_pretrained(str(merged))
    print(f"Merged HF model → {merged}")

    convert = shutil.which("convert_hf_to_gguf.py")
    if convert:
        gguf_out = out / f"pyro-coder-{quant}.gguf"
        subprocess.run(
            [convert, str(merged), "--outfile", str(gguf_out), "--outtype", quant],
            check=False,
        )
        if gguf_out.is_file():
            print(f"GGUF → {gguf_out}")
            print(f"Next: ollama create pyro-coder:7b -f {out}/Modelfile")
            return

    modelfile = out / "Modelfile"
    modelfile.write_text(
        f"# After GGUF exists: ollama create pyro-coder:7b -f {modelfile}\n"
        f"# FROM {out}/pyro-coder-{quant}.gguf\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER num_ctx 8192\n",
        encoding="utf-8",
    )
    print(f"Modelfile template → {modelfile}")
    print(
        "Install llama.cpp convert_hf_to_gguf.py on merged-hf, then:\n"
        f"  ollama create pyro-coder:7b -f {modelfile}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter for GGUF export")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory with saved adapter (from train-qlora)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        help="Target GGUF quant (q4_k_m default; Unsloth supports fast_quantized etc.)",
    )
    parser.add_argument(
        "--unsloth",
        action="store_true",
        help="Use Unsloth save_pretrained_gguf (recommended for Dynamic 2.0 path)",
    )
    args = parser.parse_args()

    out = args.out or (exports_dir() / "pyro-coder")
    if args.unsloth:
        _export_unsloth_gguf(args.adapter_dir, out, args.quant)
    else:
        _export_hf_merge(args.adapter_dir, out, args.quant)


if __name__ == "__main__":
    main()
