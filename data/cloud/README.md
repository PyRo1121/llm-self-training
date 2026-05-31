# Cloud personal data (operator)

Tier-1 personal rows for Jarvis cloud train (`make cloud-export-personal`).

Jarvis `train-cloud.sh` reads `personal-tier1.jsonl` after `git clone`.

Regenerate after new phase1/curate:

```bash
make cloud-export-personal
git add data/cloud/personal/personal-tier1.jsonl
git commit -m "Refresh cloud personal tier-1 export"
```
