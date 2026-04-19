const API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");
const BASE = API_BASE === "" ? "/api" : API_BASE;

async function parse<T>(res: Response | Promise<Response>): Promise<T> {
  const response = await res;
  if (!response.ok) {
    const t = await response.text();
    throw new Error(t || response.statusText);
  }
  return response.json() as Promise<T>;
}

async function triggerDownload(url: string, fallbackName: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) {
    const t = await response.text();
    throw new Error(t || response.statusText);
  }
  const blob = await response.blob();
  const cd = response.headers.get("Content-Disposition");
  let name = fallbackName;
  if (cd) {
    const m = /filename="?([^";]+)"?/i.exec(cd);
    if (m?.[1]) name = m[1];
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export type Processor = "auto" | "cuda" | "mps" | "cpu";
export type PipelineMode = "classify" | "ner" | "normalize" | "full";

export interface Project {
  id: string;
  name: string;
  disease_key: string;
  description: string;
}

export interface DeviceInfo {
  available: { cuda: boolean; mps: boolean; cpu: boolean };
  recommended: string;
}

export interface LastRunInfo {
  path: string | null;
  files: string[];
}

export type ExportArtifact =
  | "classification"
  | "mentions"
  | "normalized"
  | "bundle";
export type ExportFormat = "csv" | "xlsx" | "pkl";

export interface TrainingConfig {
  processor: Processor;
  base_model: string;
  learning_rate: number;
  num_train_epochs: number;
  per_device_train_batch_size: number;
  per_device_eval_batch_size: number;
  weight_decay: number;
  seed: number;
  max_length: number;
  fp16: boolean | null;
}

export interface Article {
  pmid: string;
  text: string;
  label?: number | null;
}

export const api = {
  health: () => parse<{ status: string }>(fetch(`${BASE}/health`)),

  devices: () => parse<DeviceInfo>(fetch(`${BASE}/devices`)),

  listProjects: () => parse<Project[]>(fetch(`${BASE}/projects`)),

  createProject: (body: {
    name: string;
    disease_key: string;
    description?: string;
  }) =>
    parse<Project>(
      fetch(`${BASE}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
    ),

  listModels: (projectId: string) =>
    parse<{ models: string[] }>(
      fetch(`${BASE}/projects/${projectId}/models`)
    ),

  trainRelevance: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig,
    validation_split: number
  ) =>
    parse<{ job_id: string; state: string }>(
      fetch(`${BASE}/train/${projectId}/relevance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          articles,
          config,
          validation_split,
        }),
      })
    ),

  trainKfold: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig & { n_splits: number }
  ) =>
    parse<{ job_id: string; state: string }>(
      fetch(`${BASE}/train/${projectId}/relevance/kfold`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ articles, config }),
      })
    ),

  jobStatus: (jobId: string) =>
    parse<{
      job_id: string;
      state: string;
      message: string;
      result: unknown;
    }>(fetch(`${BASE}/train/jobs/${jobId}`)),

  lastRun: (projectId: string) =>
    parse<LastRunInfo>(fetch(`${BASE}/projects/${projectId}/data/last-run`)),

  importArticles: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(
      `${BASE}/projects/${projectId}/data/import/articles`,
      { method: "POST", body: fd }
    );
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json() as Promise<{
      kind: string;
      row_count: number;
      articles: Article[];
    }>;
  },

  importMentions: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(
      `${BASE}/projects/${projectId}/data/import/mentions`,
      { method: "POST", body: fd }
    );
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json() as Promise<{
      kind: string;
      row_count: number;
      mentions: Record<string, unknown>[];
    }>;
  },

  downloadExport: (
    projectId: string,
    artifact: ExportArtifact,
    format: ExportFormat,
    fallbackName: string
  ) =>
    triggerDownload(
      `${BASE}/projects/${projectId}/data/export/${artifact}?format=${format}`,
      fallbackName
    ),

  downloadTemplate: (projectId: string, kind: "articles" | "mentions") =>
    triggerDownload(
      `${BASE}/projects/${projectId}/data/templates/${kind}`,
      kind === "articles" ? "articles_template.csv" : "mentions_template.csv"
    ),

  runPipeline: (body: {
    project_id: string;
    model_id: string;
    articles: Article[];
    mode: PipelineMode;
    processor: Processor;
    ner_model: string;
    batch_size: number;
    use_wikipedia_fallback: boolean;
    mentions_json?: Record<string, unknown>[];
  }) =>
    parse<Record<string, unknown>>(
      fetch(`${BASE}/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
    ),
};
