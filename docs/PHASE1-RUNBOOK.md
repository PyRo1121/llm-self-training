# Phase 1 runbook — data lake (May 2026)

**Exit criteria (ROADMAP):** ≥200 tier-1 curated rows · 50-row secrets+PII audit · replay buffer seeded.

**GPU:** Phase 1 ingest/curate is CPU/disk. If you run **train** or other GPU jobs on the same machine, stop **hyprwhspr** first (~5 GiB VRAM) — `train-qlora` does this automatically (`docs/PHASE2-TRAIN.md`).

**One command:**

```bash
cd "/home/pyro1121/Documents/LLM Self Training"
uv sync --package llm-dataprep --extra git
uv run --package llm-dataprep phase1 \
  --include-subagents \
  --repo "/home/pyro1121/Documents/Radar" \
  --mark-exec
```

## Pipeline steps (what `phase1` runs)

| Step | CLI | Notes |
|------|-----|-------|
| 1 | `agent-ingest` | All full/partial harnesses on disk |
| 2 | `scan-raw` | Regex per row; gitleaks **off** by default (use `--gitleaks --gitleaks-per-file` on `phase1`) |
| 3 | `curate-raw` | Session group + chunk; tier-1 gate; `style_tags` |
| 4 | `link-logs-to-diffs` | Git commit hints on curated `meta` |
| 5 | `replay-seed` | `data/replay/replay-YYYY-MM-DD.jsonl` |
| 6 | `audit-sample` | `docs/audits/phase1-audit-*.md` (50 rows) |

## Research log (tools used)

| Topic | Source | Verdict |
|-------|--------|---------|
| Presidio PII | Context7 `/microsoft/presidio` — `AnalyzerEngine().analyze(text=, language='en')` | Optional extra `[safety]` |
| Gitleaks v8 | WebSearch — `gitleaks dir` + JSON report; **not** per-row at 40k scale | `--gitleaks-per-file` in `scan-raw` |
| Codex rollouts | Local probe May 2026 | Assistant text in `output_text` blocks (not `input_text`) |
| Cursor SQLite | ROADMAP | Dual ingest deferred; JSONL is v0 source of truth |

## Policy (unchanged)

- Filter: secrets + PII only (`docs/PHASE1-FILTERS.md`)
- Do **not** filter security/gray-area/refusal topics

## Public datasets (optional, recommended)

```bash
export HF_TOKEN=hf_...   # for gated SALT-NLP/SWE-chat
uv run --package llm-dataprep public-ingest --skip-gated   # or omit flag if token set
uv run --package llm-dataprep curate-raw --no-gitleaks --no-presidio   # merges personal + public raw
```

Or: `phase1 --public --skip-gated` (ingest public before personal).

Details: [`PUBLIC-DATASETS.md`](PUBLIC-DATASETS.md).

## After Phase 1

Audit flagged rows in `docs/audits/`, then Phase 2 QLoRA or Phase 1.5 Turso per `docs/TURSO.md`.
