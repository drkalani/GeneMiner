const STORAGE_KEY = "geneminer-api-base";
const DEFAULT_API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");
const FALLBACK_API_BASE = "/api";
const API_BASE_CHANGED = "geneminer:api-base-changed";

const sanitizeBase = (value: string): string => {
  let base = (value || "").trim().replace(/\/$/, "");
  if (!base) {
    return DEFAULT_API_BASE || FALLBACK_API_BASE;
  }
  if (base.startsWith("/")) {
    return base || FALLBACK_API_BASE;
  }
  if (!/^[a-z][a-z0-9+\-.]*:\/\//i.test(base)) {
    base = `https://${base}`;
  }
  return base;
};

const getStoredApiBase = (): string => {
  if (typeof window === "undefined") {
    return sanitizeBase(DEFAULT_API_BASE);
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && stored.trim()) {
      return sanitizeBase(stored);
    }
  } catch {
    // storage may be unavailable in some environments
  }
  return sanitizeBase(DEFAULT_API_BASE);
};

let runtimeApiBase = getStoredApiBase();

const normalizePath = (path: string): string => {
  return path.startsWith("/") ? path : `/${path}`;
};

const buildApiUrl = (path: string): string => `${getApiBase()}${normalizePath(path)}`;

export const getApiBase = (): string => runtimeApiBase;

export const setApiBase = (candidate: string): string => {
  const next = sanitizeBase(candidate);
  runtimeApiBase = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
    window.dispatchEvent(new Event(API_BASE_CHANGED));
  } catch {
    // no-op if storage is not available
  }
  return next;
};

export const resetApiBase = (): string => setApiBase(DEFAULT_API_BASE);

export const apiBaseDidChange = (listener: () => void): (() => void) => {
  if (typeof window === "undefined") {
    return () => {};
  }
  const handler = () => listener();
  window.addEventListener(API_BASE_CHANGED, handler);
  return () => window.removeEventListener(API_BASE_CHANGED, handler);
};

export const pingBackend = (candidateBase?: string): Promise<{ status: string }> => {
  const targetBase = candidateBase ? sanitizeBase(candidateBase) : getApiBase();
  const url = `${targetBase}/health`;
  return parse<{ status: string }>(fetch(url));
};

export const isColabCandidate = (value: string): boolean =>
  /^(https?:\/\/)?[^/]+ngrok(app)?(-(free))?\.(io|app)/i.test(value.trim());

async function parse<T>(res: Response | Promise<Response>): Promise<T> {
  const response = await res;
  if (!response.ok) {
    const t = await response.text();
    throw new Error(t || response.statusText);
  }
  return response.json() as Promise<T>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return parse<T>(fetch(buildApiUrl(path), init));
}

async function triggerDownload(path: string, fallbackName: string): Promise<void> {
  const response = await fetch(buildApiUrl(path));
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

export interface JobRecord {
  job_id: string;
  state: "queued" | "running" | "completed" | "failed";
  message: string;
  project_id: string | null;
  created_at: string;
  result?: Record<string, unknown> | null;
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
  health: () => request<{ status: string }>("/health"),

  devices: () => request<DeviceInfo>("/devices"),

  listProjects: () => request<Project[]>("/projects"),

  createProject: (body: {
    name: string;
    disease_key: string;
    description?: string;
  }) =>
    request<Project>("/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  listModels: (projectId: string) =>
    request<{ models: string[] }>(`/projects/${projectId}/models`),

  trainRelevance: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig,
    validation_split: number
  ) =>
    request<{ job_id: string; state: string }>(`/train/${projectId}/relevance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        articles,
        config,
        validation_split,
      }),
    }),

  trainKfold: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig & { n_splits: number }
  ) =>
    request<{ job_id: string; state: string }>(`/train/${projectId}/relevance/kfold`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ articles, config }),
    }),

  jobStatus: (jobId: string) =>
    request<{
      job_id: string;
      state: string;
      message: string;
      result: unknown;
    }>(`/train/jobs/${jobId}`),

  listJobs: (projectId?: string, limit?: number) => {
    const params = new URLSearchParams();
    if (projectId) {
      params.set("project_id", projectId);
    }
    if (typeof limit === "number" && Number.isInteger(limit) && limit > 0) {
      params.set("limit", `${limit}`);
    }
    const query = params.toString();
    return request<JobRecord[]>(`/train/jobs${query ? `?${query}` : ""}`);
  },

  lastRun: (projectId: string) =>
    request<LastRunInfo>(`/projects/${projectId}/data/last-run`),

  importArticles: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(buildApiUrl(`/projects/${projectId}/data/import/articles`), {
      method: "POST",
      body: fd,
    });
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
    const res = await fetch(buildApiUrl(`/projects/${projectId}/data/import/mentions`), {
      method: "POST",
      body: fd,
    });
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
    triggerDownload(`/projects/${projectId}/data/export/${artifact}?format=${format}`, fallbackName),

  downloadTemplate: (projectId: string, kind: "articles" | "mentions") =>
    triggerDownload(
      `/projects/${projectId}/data/templates/${kind}`,
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
    request<Record<string, unknown>>("/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
