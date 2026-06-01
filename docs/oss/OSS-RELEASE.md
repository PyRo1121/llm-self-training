# Open source release checklist

Use this before making the repository public or tagging a release.

## What stays local (never commit)

Per `.gitignore`:

| Path | Contains |
|------|----------|
| `data/` | Raw logs, curated chats, train files, warehouse DB, Chroma |
| `runs/` | LoRA adapters, checkpoints, train logs |
| `exports/` | Merged weights, GGUF |
| `.env`, `.env.*` | Tokens and secrets |

**Policy:** No real agent transcripts, credentials, or personal paths in git.

## What is committed — review before publish

| Path | Risk | Action |
|------|------|--------|
| `docs/audits/phase1-audit-*.jsonl` | 400-char previews with local paths | Redact or remove before public push |
| `eval/internal/*.jsonl` | Placeholder tasks OK; real tasks may embed repo paths | Scrub if adding real eval data |
| `config/default.yaml` | Machine-specific paths in comments | OK — uses relative paths |
| `README.md`, `docs/oss/*` | Should contain no secrets | Review |

## Pre-release hygiene

```bash
# 1. Dependencies
make sync-all

# 2. Safety pipeline on any sample data you ship as examples
make sync-safety
python -m spacy download en_core_web_sm
make sanitize
make curate
make audit-sample

# 3. Review audit output
less docs/audits/phase1-audit-*.md
# Redact docs/audits/*.jsonl if paths/snippets are sensitive

# 4. Verify no secrets in tracked files
git grep -iE 'sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16}' -- ':!docs/oss'

# 5. Smoke test pipeline
make warehouse-smoke
make train-dry-run
make verify-phase15   # needs api + dashboard build
```

## Public Hugging Face data

From [PUBLIC-DATASETS.md](PUBLIC-DATASETS.md):

- Check license on each dataset card before redistribution
- Ingest is **local only** — do not commit raw HF rows
- Document `HF_TOKEN` requirement for gated sets in README
- Respect `max_rows` caps in config

## Safety policy for contributors

**Training data filters:** secrets + PII only.

**Not filtered:** security research content, refusal tone, gray-area coding topics (`filter_topics_or_refusals: false`).

**Model behavior gates** (separate from dataprep): coding eval suites only — see [CODING-SAFEGUARDS.md](CODING-SAFEGUARDS.md).

## Recommended LICENSE

**Added:** [LICENSE](../LICENSE) — **LLM Self Training Source Available License v1.0**

- Personal / non-commercial use and inspection allowed
- **No** commercial use, public forks, or derivative redistribution without written permission
- Not OSI-approved “open source” (MIT/Apache) — intentional

Commercial licensing: olen@latham.cloud

## Recommended CONTRIBUTING.md

See [CONTRIBUTING.md](../../CONTRIBUTING.md) at repo root.

## Audit artifacts

`docs/audits/*.jsonl` is **gitignored** (local path previews). Operators generate via `make audit-sample`. See [docs/audits/README.md](../audits/README.md).

## Post-release operator notes

- Cloners get **empty** `data/`, `runs/`, `exports/` until pipelines run
- `unsloth-zoo` may need manual install after `uv sync` (datasets version conflict):
  ```bash
  uv pip install "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
  ```
- flash-attn requires source build on torch 2.12+cu130:
  ```bash
  MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
  ```

## Planned promote-time sweep (not yet in code)

`docs/archive/PLAN.md` specifies TruffleHog no-verify at promote audit — run manually until implemented.

## Related

- [USER-GUIDE.md](USER-GUIDE.md) — sanitize workflow
- [OSS-RELEASE.md](OSS-RELEASE.md) — release safety checklist
- [PRODUCT.md](PRODUCT.md)
