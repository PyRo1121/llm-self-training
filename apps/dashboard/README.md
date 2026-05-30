# Operator dashboard

Bun + Vite + React. Talks to `apps/api` only (no direct DB).

```bash
# Terminal 1
uv run --package llm-api llm-api

# Terminal 2
cd apps/dashboard && bun install && bun dev
```

Open http://127.0.0.1:5173

Phase 1.5 adds shadcn + TanStack Table per `ROADMAP.md`; this scaffold proves API wiring.
