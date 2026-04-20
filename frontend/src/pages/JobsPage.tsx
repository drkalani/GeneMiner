import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type JobRecord } from "../api";

type JobState = JobRecord["state"];
type JobStateFilter = JobState | "all";

const MAX_ROWS = 150;

const stateColor: Record<JobState, string> = {
  queued: "var(--warning)",
  running: "var(--accent)",
  completed: "var(--success)",
  failed: "var(--danger)",
};

const stateLabel: Record<JobState, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};
const statePriority: Record<JobState, number> = {
  running: 0,
  queued: 1,
  completed: 2,
  failed: 3,
};

export function JobsPage() {
  const [searchParams] = useSearchParams();
  const projectFilter = searchParams.get("projectId") ?? "";
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [stateFilter, setStateFilter] = useState<JobStateFilter>("all");
  const [query, setQuery] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [copyNotice, setCopyNotice] = useState<string | null>(null);

  const loadJobs = async () => {
    if (!refreshing) {
      setRefreshing(true);
    }
    try {
      const r = await api.listJobs(projectFilter || undefined, MAX_ROWS);
      const sorted = r.slice().sort((a, b) => {
        const p = statePriority[a.state] - statePriority[b.state];
        if (p !== 0) {
          return p;
        }
        if (a.created_at === b.created_at) {
          return 0;
        }
        return a.created_at > b.created_at ? -1 : 1;
      });
      setJobs(sorted);
      setErr(null);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    void loadJobs();
    if (!autoRefresh) {
      return;
    }
    const timer = setInterval(() => {
      void loadJobs();
    }, 4000);
    return () => clearInterval(timer);
  }, [autoRefresh, projectFilter]);

  const filteredJobs = jobs.filter((j) => {
    if (stateFilter !== "all" && j.state !== stateFilter) {
      return false;
    }
    if (!query.trim()) return true;
    const needle = query.trim().toLowerCase();
    return (
      j.job_id.toLowerCase().includes(needle) ||
      (j.project_id || "").toLowerCase().includes(needle) ||
      j.message.toLowerCase().includes(needle)
    );
  });

  const allInProgress = jobs.filter((j) => j.state === "queued" || j.state === "running").length;
  const visibleCount = filteredJobs.length;
  const counts = {
    queued: jobs.filter((j) => j.state === "queued").length,
    running: jobs.filter((j) => j.state === "running").length,
    completed: jobs.filter((j) => j.state === "completed").length,
    failed: jobs.filter((j) => j.state === "failed").length,
  };

  const summaryText = JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      scope: projectFilter || "all_projects",
      filters: {
        state: stateFilter,
        query: query.trim() || null,
      },
      totals: {
        all: jobs.length,
        filtered: visibleCount,
        in_progress: allInProgress,
        queued: counts.queued,
        running: counts.running,
        completed: counts.completed,
        failed: counts.failed,
      },
      jobs: filteredJobs.map((job) => ({
        job_id: job.job_id,
        state: job.state,
        message: job.message || "",
        project_id: job.project_id,
        created_at: job.created_at,
      })),
    },
    null,
    2
  );

  const copySummary = async () => {
    try {
      if (!navigator?.clipboard?.writeText) {
        throw new Error("Clipboard API is not available in this browser.");
      }
      await navigator.clipboard.writeText(summaryText);
      setCopyNotice("Summary copied.");
      setTimeout(() => setCopyNotice(null), 1800);
    } catch (e) {
      setCopyNotice((e as Error).message || "Unable to copy summary.");
      setTimeout(() => setCopyNotice(null), 1800);
    }
  };

  return (
    <div>
      <p>
        <Link to="/">← Back to overview</Link>
      </p>
      <h1>Job center</h1>
      <p style={{ color: "var(--muted)" }}>
        {projectFilter
          ? `Project-scoped view for ${projectFilter}.`
          : "Track all active and recent jobs from all projects."}
      </p>

      <div className="toolbar">
        <button className="btn" onClick={() => void loadJobs()} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh now"}
        </button>
        <label style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem", color: "var(--muted)" }}>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto refresh
        </label>
        <select
          value={stateFilter}
          onChange={(e) => setStateFilter(e.target.value as JobStateFilter)}
          style={{ maxWidth: 170 }}
        >
          <option value="all">All states</option>
          <option value="queued">Queued</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
        <input
          type="search"
          placeholder="Search job id / project / message"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ maxWidth: 220 }}
        />
        <span className="tag">
          Showing {visibleCount} / {jobs.length}
        </span>
        <button className="btn" type="button" onClick={() => void copySummary()} disabled={jobs.length === 0 || refreshing}>
          Copy summary
        </button>
        {copyNotice && <span className="tag">{copyNotice}</span>}
        {allInProgress > 0 && (
          <span className="tag" style={{ color: "var(--warning)" }}>
            {allInProgress} job(s) in progress
          </span>
        )}
        <Link to="/projects" className="btn">
          Open projects
        </Link>
      </div>

      {err && (
        <div className="card" style={{ borderColor: "var(--danger)" }}>
          {err}
        </div>
      )}

      {loading && <p style={{ color: "var(--muted)" }}>Loading jobs…</p>}
      {!loading && filteredJobs.length === 0 && (
        <p style={{ color: "var(--muted)" }}>
          {jobs.length === 0 ? "No jobs yet." : "No jobs match the selected filters."}
        </p>
      )}

      <div style={{ display: "grid", gap: "0.75rem" }}>
        {filteredJobs.map((job) => (
          <div className="card" key={job.job_id}>
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
              <span className="tag" style={{ color: stateColor[job.state], borderColor: stateColor[job.state] }}>
                {stateLabel[job.state]}
              </span>
              <strong>{job.job_id}</strong>
            </div>
            <p style={{ margin: "0.55rem 0 0", color: "var(--muted)" }}>
              Project:{" "}
              {job.project_id ? (
                <Link to={`/projects/${job.project_id}`}>{job.project_id}</Link>
              ) : (
                "n/a"
              )}
            </p>
            <p style={{ margin: "0.35rem 0 0", color: "var(--muted)" }}>
              Message: {job.message || "No details"}
            </p>
            <p style={{ margin: "0.35rem 0 0", color: "var(--muted)", fontSize: "0.85rem" }}>
              Started: {job.created_at || "unknown"}
            </p>
            {job.result && (
              <div style={{ marginTop: "0.75rem" }}>
                <details>
                  <summary style={{ color: "var(--muted)", cursor: "pointer" }}>Result</summary>
                  <pre
                    className="mono"
                    style={{
                      marginTop: "0.5rem",
                      padding: "0.75rem",
                      background: "var(--bg)",
                      borderRadius: 8,
                      overflow: "auto",
                      maxHeight: "220px",
                    }}
                  >
                    {JSON.stringify(job.result, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

