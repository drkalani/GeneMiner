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
