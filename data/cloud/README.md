# Cloud personal data (Vast / Jarvis)

After `git clone`, training uses everything under this folder — no local harness paths on the cloud box.

| Path | Purpose |
|------|---------|
| `personal-tier1.jsonl` | All personal tier-1 rows (merged) |
| `harnesses/*.jsonl` | Per-agent shards (cursor, codex, claude_code, …) |
| `manifest.json` | Row counts + harness index |

Regenerate from your machine (has `~/.cursor`, `~/.codex`, etc.):

```bash
make cloud-export-all    # ingest all harnesses → curate → export
git add data/cloud/personal/
git commit -m "Refresh cloud personal export"
git push
```

Vast `train-cloud.sh` merges `personal-tier1.jsonl` + `harnesses/*.jsonl` on boot.

HF token: `config/cloud.env` in repo + passed to Vast Docker env on `make cloud-vast`.
