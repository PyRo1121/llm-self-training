# Phase 1 audit outputs (local only)

> **Note:** JSONL previews are gitignored; MD summaries may be committed if redacted.

`audit-sample` writes operator review artifacts here:

- `phase1-audit-YYYY-MM-DD.md` — summary table (safe to commit if no sensitive previews)
- `phase1-audit-YYYY-MM-DD.jsonl` — **gitignored** — 400-char text previews may contain local paths or transcript snippets

Generate locally:

```bash
make audit-sample
# or after curate:
uv run --package llm-dataprep audit-sample \
  --curated data/curated/curated-YYYY-MM-DD.jsonl
```

Review flagged rows before training. Do not commit JSONL previews to a public repo.
