# Phase 1 — Safety filters (step-by-step)

**ROADMAP item:** `filters.py` — gitleaks + Presidio/GLiNER on every row.  
**Order:** filters → tier-1 gate → curated JSONL (Turso warehouse is Phase 1.5).

## What we filter vs what we do not

| In scope | Out of scope (never add to dataprep) |
|----------|--------------------------------------|
| API keys, tokens, private keys (gitleaks + regex) | “Hacking,” exploits, security research, jailbreaks |
| PII you do not want in weights (Presidio) | Refusal tone, policy violations, “harmful” topics |
| Accidental paste of *your* credentials | Blocking honest-but-edgy coding help |

**Quarantine** in this repo means **secrets/PII or bad eval rows** — not “the model talked about something naughty.”

Training goal: **less base-model refusal**, not more — SFT on *your* accepted transcripts (including security work) plus a direct system prompt at inference. That is separate from `scan-raw`.

**Operator policy** (`config/default.yaml` → `curation`): secrets/PII yes; topic/refusal filters no. Gray-area technical content (security research, edgy tooling, etc.) stays in the lake. Obvious illegal *acts* are not keyword-scanned in dataprep — that boundary is runtime/operator, not automated row drops.

## MCP / research log (this step)

| Tool | Used for |
|------|----------|
| **Context7** | `/microsoft/presidio` — `AnalyzerEngine`, `analyze(text=..., language='en')` |
| **WebSearch** | gitleaks v8: `gitleaks dir`, `gitleaks stdin`, `--report-format json` |
| **Exa** | Rate-limited — skipped |

**gitleaks (CLI, not pip):** Scan via `gitleaks dir` (v8.19+) with `--report-format json` ([README](https://github.com/gitleaks/gitleaks/blob/master/README.md)). `phase1` auto-enables per-file gitleaks when `gitleaks` is on `PATH` (`--no-gitleaks` to opt out). `curate-raw` skips sessions listed in `safety-failures-*.jsonl` by default (`--no-honor-safety-failures` to opt out).

**Presidio (pip):** Optional extra `llm-dataprep[safety]`. Requires spaCy model (`python -m spacy download en_core_web_sm` or `en_core_web_lg`). Heavy — lazy-import in `filters.py`.

**GLiNER:** Deferred (PLAN allows Presidio first).

## Implementation slices

1. [x] `filters.py` v0 — regex + optional gitleaks subprocess + optional Presidio
2. [x] CLI `scan-raw` — `uv run --package llm-dataprep scan-raw` → `safety-failures-*.jsonl`
3. [x] `tier1.py` — PLAN gate (`accepted` + exec/verify + safety pass; bootstrap allows unknown exec/verify)
4. [x] `curate-raw` — `uv run --package llm-dataprep curate-raw` → `data/curated/*.jsonl` (OpenAI `messages[]` + `meta`)

## Install (when running safety)

```bash
# gitleaks (Arch example)
sudo pacman -S gitleaks

uv sync --package llm-dataprep --extra safety
python -m spacy download en_core_web_sm
```
