# Coding safeguards (inform, don't refuse)

**Policy:** Local Pyro Coder — no content nanny. **Coding quality** gates only. Illegal/dodgy requests get a **short factual heads-up**, then technical answers. User decides.

## Stack pieces

| Piece | Path / command |
|-------|----------------|
| Abliterated HF base | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` |
| Config | `config/default.yaml` → `train.decensor` |
| Cache weights (CPU/disk, OK during GPU train) | `uv run --package llm-train train-download-base --decensor` |
| Ollama reference (smoke) | `ollama pull huihui_ai/qwen2.5-coder-abliterate:7b` |
| System prompt | `config/modelfiles/pyro-coder-inform.modelfile` |
| Optional SFT slice | `data/train/inform-dont-refuse.jsonl` |

## Order of operations

1. **Finish** current promote on aligned base (`pyro-coder-promote-v2`) — capability baseline.
2. **Download** abliterated base while GPU busy: `train-download-base --decensor`.
3. **Train** personal mix on abliterated base:
   ```bash
   uv run --package llm-train train-preflight --decensor
   uv run --package llm-train train-qlora --decensor --max-steps 400 --run-name pyro-coder-uncensored-v1
   ```
4. **Export** → point Modelfile `FROM` at merged HF → `ollama create pyro-coder:7b -f config/modelfiles/pyro-coder-inform.modelfile`.
5. **Promote gate:** `run-eval --strict` on **coding** suites only — not compliance refusals.

## What we do not gate on

- Refusal to answer “sensitive” topics
- Political / vendor safety scores
- JailbreakBench-style block lists

## What we do gate on

- `diff_apply`, `style`, `debug` (real tasks in `eval/internal/`)
- Secrets not committed in training data (`curation.filter_secrets_and_pii`)
- Optional manual spot-check: dodgy prompt → warns + still helps

## Optional: merge inform slice

Concatenate or second short run on `inform-dont-refuse.jsonl` if abliterated base still hard-refuses without context. Keep slice small (hundreds of rows), your voice.

## Heretic (DIY abliterate)

If pre-abliterated weights drift from upstream: [p-e-w/heretic](https://github.com/p-e-w/heretic) on stock Qwen — run **after** GPU train finishes, not during.
