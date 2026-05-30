import { useEffect, useState } from "react";

type Overview = {
  curated_total?: number;
  curated_by_tier?: Record<string, number>;
  quarantine_events?: number;
  rag_sources_indexed?: number;
  ingest_files?: number;
  source_registry_entries?: number;
  warehouse_driver?: string;
  training_runs_in_db?: number;
  training_manifests?: number;
  latest_ingest_run?: {
    run_id: string;
    pipeline: string;
    status: string;
    total_rows: number;
  } | null;
  runs_on_disk?: { run_name: string; has_adapter: boolean }[];
};

type DataLake = {
  by_data_source?: Record<string, number>;
  top_harnesses?: { harness: string; count: number }[];
  public_datasets?: { dataset: string; count: number }[];
  raw_files?: { path: string; harness: string | null; row_count: number }[];
};

type QuarantineRow = {
  id: number;
  curated_id: string;
  reason: string;
  harness: string | null;
  created_at: string;
};

type RagStatus = {
  chroma?: { count?: number };
};

type TrainingRunDb = {
  run_name: string;
  status: string;
  base_model?: string;
  adapter_path?: string;
  train_rows?: number;
  started_at?: string;
  finished_at?: string;
};

type TrainingRuns = {
  database?: TrainingRunDb[];
  disk?: { run_name: string; has_adapter: boolean; adapter_path?: string }[];
};

type Tab = "overview" | "datalake" | "training";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [datalake, setDatalake] = useState<DataLake | null>(null);
  const [quarantine, setQuarantine] = useState<QuarantineRow[]>([]);
  const [training, setTraining] = useState<TrainingRuns | null>(null);
  const [rag, setRag] = useState<RagStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchJson<Overview>("/api/v1/overview"),
      fetchJson<DataLake>("/api/v1/datalake/summary"),
      fetchJson<QuarantineRow[]>("/api/v1/datalake/quarantine?limit=20"),
      fetchJson<TrainingRuns>("/api/v1/training/runs"),
      fetchJson<RagStatus>("/api/v1/rag/status"),
    ])
      .then(([o, d, q, t, r]) => {
        setOverview(o);
        setDatalake(d);
        setQuarantine(q);
        setTraining(t);
        setRag(r);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const chromaCount = rag?.chroma?.count;

  return (
    <main>
      <header className="header">
        <h1>LLM Self Training — Control Plane</h1>
        <nav className="tabs">
          <button
            type="button"
            className={tab === "overview" ? "tab active" : "tab"}
            onClick={() => setTab("overview")}
          >
            Overview
          </button>
          <button
            type="button"
            className={tab === "datalake" ? "tab active" : "tab"}
            onClick={() => setTab("datalake")}
          >
            Data Lake
          </button>
          <button
            type="button"
            className={tab === "training" ? "tab active" : "tab"}
            onClick={() => setTab("training")}
          >
            Training
          </button>
        </nav>
      </header>

      {error && (
        <p className="error">
          API: {error} — run <code>uv run --package llm-api llm-api</code>
        </p>
      )}

      {overview?.warehouse_driver && (
        <p className="muted driver">{overview.warehouse_driver}</p>
      )}

      {tab === "overview" && (
        <div className="grid">
          <section className="card">
            <h2>Curated examples</h2>
            <div className="stat">{overview?.curated_total?.toLocaleString() ?? "—"}</div>
            <p className="muted">Tier-1 rows in warehouse</p>
          </section>
          <section className="card">
            <h2>Raw files</h2>
            <div className="stat">{overview?.ingest_files ?? "—"}</div>
            <p className="muted">Indexed in ingest_files</p>
          </section>
          <section className="card">
            <h2>Registry</h2>
            <div className="stat">{overview?.source_registry_entries ?? "—"}</div>
            <p className="muted">source_registry entries</p>
          </section>
          <section className="card">
            <h2>Manifests</h2>
            <div className="stat">{overview?.training_manifests ?? "—"}</div>
            <p className="muted">Training mix manifests</p>
          </section>
          <section className="card">
            <h2>Quarantine</h2>
            <div className="stat">{overview?.quarantine_events ?? 0}</div>
            <p className="muted">Quarantine events</p>
          </section>
          <section className="card">
            <h2>RAG (Chroma)</h2>
            <div className="stat">{chromaCount ?? "—"}</div>
            <p className="muted">
              Chunks · {overview?.rag_sources_indexed ?? 0} sources in warehouse
            </p>
          </section>
          <section className="card wide">
            <h2>Latest ingest run</h2>
            {overview?.latest_ingest_run ? (
              <pre>{JSON.stringify(overview.latest_ingest_run, null, 2)}</pre>
            ) : (
              <p className="muted">No ingest_runs yet — run warehouse-load</p>
            )}
          </section>
          <section className="card wide">
            <h2>Training runs</h2>
            <ul className="list">
              {(training?.disk ?? overview?.runs_on_disk ?? []).slice(0, 8).map((r) => (
                <li key={r.run_name}>
                  {r.run_name}
                  {"has_adapter" in r && r.has_adapter ? " ✓ adapter" : ""}
                </li>
              ))}
            </ul>
            <p className="muted">{overview?.training_runs_in_db ?? 0} runs in DB</p>
          </section>
        </div>
      )}

      {tab === "training" && (
        <div className="grid">
          <section className="card wide">
            <h2>Runs (warehouse)</h2>
            {(training?.database ?? []).length === 0 ? (
              <p className="muted">No training_runs in DB — run train-register</p>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Status</th>
                    <th>Rows</th>
                    <th>Adapter</th>
                  </tr>
                </thead>
                <tbody>
                  {(training?.database ?? []).map((r) => (
                    <tr key={r.run_name}>
                      <td className="mono">{r.run_name}</td>
                      <td>{r.status}</td>
                      <td>{r.train_rows ?? "—"}</td>
                      <td className="mono">
                        {r.adapter_path ? "✓" : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
          <section className="card wide">
            <h2>Runs (disk)</h2>
            <ul className="list">
              {(training?.disk ?? []).map((r) => (
                <li key={r.run_name}>
                  {r.run_name}
                  {r.has_adapter ? " ✓ adapter" : ""}
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}

      {tab === "datalake" && (
        <>
          <div className="grid">
            <section className="card">
              <h2>Personal vs public</h2>
              <pre>{JSON.stringify(datalake?.by_data_source ?? {}, null, 2)}</pre>
            </section>
            <section className="card">
              <h2>Top harnesses</h2>
              <ul className="list">
                {(datalake?.top_harnesses ?? []).slice(0, 12).map((h) => (
                  <li key={h.harness}>
                    {h.harness}: {h.count.toLocaleString()}
                  </li>
                ))}
              </ul>
            </section>
            <section className="card wide">
              <h2>Public datasets (top)</h2>
              <ul className="list cols">
                {(datalake?.public_datasets ?? []).map((d) => (
                  <li key={d.dataset}>
                    {d.dataset}: {d.count.toLocaleString()}
                  </li>
                ))}
              </ul>
            </section>
          </div>
          <section className="card wide section-gap">
            <h2>Raw JSONL files</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Harness</th>
                  <th>Rows</th>
                </tr>
              </thead>
              <tbody>
                {(datalake?.raw_files ?? []).map((f) => (
                  <tr key={f.path}>
                    <td className="mono">{f.path.split("/").slice(-1)[0]}</td>
                    <td>{f.harness ?? "—"}</td>
                    <td>{f.row_count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section className="card wide section-gap">
            <h2>Recent quarantine</h2>
            {quarantine.length === 0 ? (
              <p className="muted">No quarantine events</p>
            ) : (
              <ul className="list">
                {quarantine.map((q) => (
                  <li key={q.id}>
                    {q.curated_id.slice(0, 12)}… — {q.reason} ({q.harness ?? "?"})
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </main>
  );
}
