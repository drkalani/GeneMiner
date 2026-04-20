import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  type Article,
  type LastRunInfo,
  type DeviceInfo,
  type PipelineMode,
  type Processor,
  type TrainingConfig,
  apiBaseDidChange,
  getApiBase,
  pingBackend,
  setApiBase,
} from "../api";

const defaultTrainConfig = (): TrainingConfig => ({
  processor: "auto",
  base_model: "dmis-lab/biobert-v1.1",
  learning_rate: 2e-5,
  num_train_epochs: 4,
  per_device_train_batch_size: 16,
  per_device_eval_batch_size: 16,
  weight_decay: 0.01,
  seed: 42,
  max_length: 512,
  fp16: null,
});

const mpsTrainPreset = (): TrainingConfig => ({
  processor: "mps",
  base_model: "dmis-lab/biobert-v1.1",
  learning_rate: 2e-5,
  num_train_epochs: 3,
  per_device_train_batch_size: 4,
  per_device_eval_batch_size: 4,
  weight_decay: 0.01,
  seed: 42,
  max_length: 256,
  fp16: false,
});

const exampleArticlesJson = `[
  {"pmid":"10000001","title":"Gene regulation in DKD","text":"TGF-beta signaling in diabetic nephropathy and kidney fibrosis.","label":1},
  {"pmid":"10000002","text":"Weather patterns in coastal regions unrelated to nephrology.","label":0}
]`;
const examplePipelineArticlesJson = `[
  {"pmid":"10000001","title":"Gene regulation in DKD","text":"TGF-beta signaling in diabetic nephropathy and kidney fibrosis."},
  {"pmid":"10000002","text":"Weather patterns in coastal regions unrelated to nephrology."}
]`;

const exampleComparePrimary = `[
  {"pmid":"10000001","label":1},
  {"pmid":"10000002","label":0}
]`;
const exampleCompareLitSuggest = `[
  {"pmid":"10000001","score":0.82},
  {"pmid":"10000002","score":0.31}
]`;

type ClassificationRow = {
  pmid?: string;
  text?: string;
  relevant?: number;
  relevance_prob?: number;
};

type WorkspaceStep = "backend" | "train" | "integrations" | "pipeline";

const toNumberOrNull = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const getClassificationRows = (
  result: Record<string, unknown>
): ClassificationRow[] => {
  if (result.kind === "full") {
    const fullRows = result.classification;
    if (Array.isArray(fullRows)) {
      return fullRows as ClassificationRow[];
    }
    return [];
  }
  if (result.kind === "classification") {
    const rows = result.rows;
    if (Array.isArray(rows)) {
      return rows as ClassificationRow[];
    }
  }
  return [];
};

export function ProjectWorkspace() {
  const { id } = useParams<{ id: string }>();
  const projectId = id ?? "";

  const [models, setModels] = useState<string[]>([]);
  const [devices, setDevices] = useState<string | null>(null);
  const [trainCfg, setTrainCfg] = useState<TrainingConfig>(defaultTrainConfig);
  const [valSplit, setValSplit] = useState(0.2);
  const [kfoldSplits, setKfoldSplits] = useState(5);
  const [articlesJson, setArticlesJson] = useState(exampleArticlesJson);
  const [backendUrl, setBackendUrl] = useState(getApiBase());
  const [backendStatus, setBackendStatus] = useState<
    "pending" | "checking" | "connected" | "failed"
  >("pending");
  const [backendStatusText, setBackendStatusText] = useState("Not checked");
  const [backendMessage, setBackendMessage] = useState(
    "Test backend before running training or pipeline jobs."
  );
  const [backendDevices, setBackendDevices] = useState<DeviceInfo | null>(null);
  const [backendDevicesStatus, setBackendDevicesStatus] = useState<
    "pending" | "checking" | "done" | "error"
  >("pending");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobPoll, setJobPoll] = useState<unknown>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeStep, setActiveStep] = useState<WorkspaceStep>("backend");

  const [modelId, setModelId] = useState("");
  const [mode, setMode] = useState<PipelineMode>("full");
  const [pipeJson, setPipeJson] = useState(exampleArticlesJson);
  const [processor, setProcessor] = useState<Processor>("auto");
  const [nerModel, setNerModel] = useState("pruas/BENT-PubMedBERT-NER-Gene");
  const [pipeBatchSize, setPipeBatchSize] = useState(4);
  const [pipeUseWikipedia, setPipeUseWikipedia] = useState(true);
  const [relevanceThreshold, setRelevanceThreshold] = useState(0.5);
  const [pipeResult, setPipeResult] = useState<Record<string, unknown> | null>(
    null
  );
  const [normJson, setNormJson] = useState(
    `[{"pmid":"1","mention":"TGFB1","start":0,"end":5}]`
  );
  const [lastRunInfo, setLastRunInfo] = useState<LastRunInfo | null>(null);
  const trainFileRef = useRef<HTMLInputElement>(null);
  const pipeFileRef = useRef<HTMLInputElement>(null);
  const mentionFileRef = useRef<HTMLInputElement>(null);
  const litsuggestFileRef = useRef<HTMLInputElement>(null);

  const [pubmedEmail, setPubmedEmail] = useState("");
  const [pubmedQuery, setPubmedQuery] = useState(
    '("diabetic kidney disease"[Title/Abstract]) AND 2020:2024[dp]'
  );
  const [pubmedMax, setPubmedMax] = useState(50);
  const [pubmedMinAbstract, setPubmedMinAbstract] = useState(0);
  const [pubmedResult, setPubmedResult] = useState<{
    articles: { pmid: string; title: string; text: string }[];
    row_count: number;
    queried_id_count: number;
    search_total_estimate: number | null;
  } | null>(null);
  const [pubmedBusy, setPubmedBusy] = useState(false);

  const [comparePrimaryJson, setComparePrimaryJson] = useState(exampleComparePrimary);
  const [compareLitJson, setCompareLitJson] = useState(exampleCompareLitSuggest);
  const [compareThreshold, setCompareThreshold] = useState(0.5);
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null);
  const [compareBusy, setCompareBusy] = useState(false);

  const isBackendConnected = backendStatus === "connected";

  const jumpToStep = (step: WorkspaceStep) => {
    setActiveStep(step);
    const node = document.getElementById(`workspace-${step}`);
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const refreshLastRun = () => {
    if (!projectId) return;
    api
      .lastRun(projectId)
      .then(setLastRunInfo)
      .catch(() => setLastRunInfo(null));
  };

  const refreshModels = () => {
    if (!projectId) return;
    api.listModels(projectId).then((r) => {
      setModels(r.models);
      if (r.models.length && !modelId) {
        setModelId(r.models[r.models.length - 1]!);
      }
    });
  };

  useEffect(() => {
    api.devices().then((d) => {
      setDevices(
        `Recommended: ${d.recommended} · CUDA ${d.available.cuda ? "on" : "off"
        } · MPS ${d.available.mps ? "on" : "off"}`
      );
    });
  }, []);

  useEffect(() => {
    refreshModels();
  }, [projectId]);

  useEffect(() => {
    refreshLastRun();
  }, [projectId]);

  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(() => {
      api.jobStatus(jobId).then((j) => {
        setJobPoll(j);
        if (j.state === "completed" || j.state === "failed") {
          clearInterval(t);
          refreshModels();
        }
      });
    }, 1500);
    return () => clearInterval(t);
  }, [jobId]);

  const parseArticles = (): Article[] => {
    const raw = JSON.parse(articlesJson) as Article[];
    if (!Array.isArray(raw)) throw new Error("Expected JSON array");
    return raw.map((a) => {
      const titleStr =
        a.title !== undefined && a.title !== null && String(a.title).trim() !== ""
          ? String(a.title).trim()
          : undefined;
      return {
        pmid: String(a.pmid),
        text: String(a.text),
        ...(titleStr !== undefined ? { title: titleStr } : {}),
        label:
          a.label === undefined || a.label === null ? undefined : Number(a.label),
      };
    });
  };

  const train = async (kfold: boolean) => {
    setErr(null);
    setBusy(true);
    try {
      const articles = parseArticles();
      const cfg = { ...trainCfg, n_splits: kfoldSplits };
      const res = kfold
        ? await api.trainKfold(projectId, articles, cfg)
        : await api.trainRelevance(projectId, articles, trainCfg, valSplit);
      setJobId(res.job_id);
      setJobPoll(null);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runPipe = async () => {
    setErr(null);
    setBusy(true);
    setPipeResult(null);
    try {
      const articles = JSON.parse(pipeJson) as Article[];
      const payload: Parameters<typeof api.runPipeline>[0] = {
        project_id: projectId,
        model_id: modelId,
        articles: articles.map((a) => {
          const base = {
            pmid: String(a.pmid),
            text: String(a.text),
            label:
              a.label === undefined || a.label === null
                ? undefined
                : Number(a.label),
          };
          const titleStr =
            a.title !== undefined &&
            a.title !== null &&
            String(a.title).trim() !== ""
              ? String(a.title).trim()
              : undefined;
          return titleStr !== undefined ? { ...base, title: titleStr } : base;
        }),
        mode,
        processor,
        ner_model: nerModel,
      batch_size: pipeBatchSize,
      use_wikipedia_fallback: pipeUseWikipedia,
      };
      if (mode === "normalize") {
        payload.mentions_json = JSON.parse(normJson) as Record<
          string,
          unknown
        >[];
      }
      const out = await api.runPipeline(payload);
      setPipeResult(out);
      refreshLastRun();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runPubmedFetch = async () => {
    if (!projectId) return;
    if (!pubmedEmail.trim() || !pubmedEmail.includes("@")) {
      setErr("Enter a valid email (required by NCBI Entrez).");
      return;
    }
    setErr(null);
    setPubmedBusy(true);
    setPubmedResult(null);
    try {
      const r = await api.pubmedFetch(projectId, {
        email: pubmedEmail.trim(),
        query: pubmedQuery.trim() || undefined,
        max_results: pubmedMax,
        min_abstract_chars: pubmedMinAbstract,
      });
      setPubmedResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setPubmedBusy(false);
    }
  };

  const copyPubmedToPipelineJson = () => {
    if (!pubmedResult?.articles?.length) return;
    const lines = pubmedResult.articles.map((a) => {
      const row: Record<string, string> = {
        pmid: a.pmid,
        text: a.text || "",
      };
      if (a.title?.trim()) row.title = a.title.trim();
      return row;
    });
    setPipeJson(JSON.stringify(lines, null, 2));
    jumpToStep("pipeline");
  };

  const runLitCompare = async () => {
    setErr(null);
    setCompareBusy(true);
    setCompareResult(null);
    try {
      const primary = JSON.parse(comparePrimaryJson) as Record<string, unknown>[];
      const litsuggest = JSON.parse(compareLitJson) as Record<string, unknown>[];
      if (!Array.isArray(primary) || !Array.isArray(litsuggest)) {
        throw new Error("Expected two JSON arrays");
      }
      const r = await api.compareLitSuggest(projectId, {
        primary,
        litsuggest,
        score_threshold: compareThreshold,
      });
      setCompareResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setCompareBusy(false);
    }
  };

  const onLitSuggestFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setErr(null);
    setCompareBusy(true);
    try {
      const r = await api.importLitSuggestScores(projectId, f);
      setCompareLitJson(JSON.stringify(r.litsuggest, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setCompareBusy(false);
    }
  };

  const applyTrainPreset = () => {
    setTrainCfg(mpsTrainPreset());
    setValSplit(0.2);
    setKfoldSplits(5);
    setArticlesJson(exampleArticlesJson);
  };

  const onTrainFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.importArticles(projectId, f);
      setArticlesJson(JSON.stringify(r.articles, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onPipeArticlesFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.importArticles(projectId, f);
      setPipeJson(JSON.stringify(r.articles, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onMentionsFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.importMentions(projectId, f);
      setNormJson(JSON.stringify(r.mentions, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runBackendCheck = async (candidate?: string) => {
    setBackendStatus("checking");
    setBackendStatusText("Checking...");
    try {
      await pingBackend(candidate);
      if (candidate !== undefined) {
        setBackendUrl(setApiBase(candidate));
      }
      setBackendStatus("connected");
      setBackendStatusText("Connected");
      setBackendMessage(
        "Backend connected. Run system check, then start training or pipeline."
      );
      setErr(null);
    } catch (err) {
      setBackendStatus("failed");
      setBackendStatusText((err as Error).message || "Unable to connect");
      setBackendMessage("Save backend URL here and click save, then retry the check.");
      setBackendDevicesStatus("pending");
      setBackendDevices(null);
    }
  };

  const runSystemCheck = async () => {
    setBackendDevicesStatus("checking");
    try {
      const d = await api.devices();
      setBackendDevices(d);
      setBackendDevicesStatus("done");
    } catch (err) {
      setBackendDevicesStatus("error");
      setErr((err as Error).message);
      setBackendMessage((err as Error).message || "System check failed.");
    }
  };

  useEffect(() => {
    const sync = () => {
      const next = getApiBase();
      setBackendUrl(next);
      void runBackendCheck();
    };
    sync();
    const stop = apiBaseDidChange(sync);
    return stop;
  }, []);

  const applyPipelinePreset = () => {
    setProcessor("mps");
    setNerModel("pruas/BENT-PubMedBERT-NER-Gene");
    setPipeBatchSize(4);
    setPipeUseWikipedia(true);
    setMode("full");
    setRelevanceThreshold(0.5);
    setPipeJson(examplePipelineArticlesJson);
  };

  const classificationRows = pipeResult ? getClassificationRows(pipeResult) : [];
  const filteredClassificationRows = classificationRows.filter((row) => {
    const prob = toNumberOrNull(row.relevance_prob);
    if (prob === null) return false;
    return prob >= relevanceThreshold;
  });

  return (
    <div>
      <p>
        <Link to="/projects">← Projects</Link>
      </p>
      <h1>Project workspace</h1>
      <div className="steps">
        <a
          href="#workspace-backend"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("backend");
          }}
          className={activeStep === "backend" ? "step-pill active" : "step-pill"}
        >
          1 · Backend
        </a>
        <a
          href="#workspace-train"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("train");
          }}
          className={activeStep === "train" ? "step-pill active" : "step-pill"}
        >
          2 · Train
        </a>
        <a
          href="#workspace-integrations"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("integrations");
          }}
          className={activeStep === "integrations" ? "step-pill active" : "step-pill"}
        >
          3 · PubMed & LitSuggest
        </a>
        <a
          href="#workspace-pipeline"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("pipeline");
          }}
          className={activeStep === "pipeline" ? "step-pill active" : "step-pill"}
        >
          4 · Pipeline
        </a>
      </div>
      <div className="toolbar" style={{ marginBottom: "1rem" }}>
        <Link to={`/jobs${projectId ? `?projectId=${projectId}` : ""}`} className="btn btn-primary">
          Open job center
        </Link>
        {!isBackendConnected && (
          <p style={{ margin: 0, color: "var(--warning)" }}>
            Backend is not connected. Connect first before training/pipeline.
          </p>
        )}
      </div>
      <div id="workspace-backend" className="card">
        <h3 style={{ marginTop: 0 }}>Backend status</h3>
        <p style={{ marginTop: 0, color: "var(--muted)" }}>{backendMessage}</p>
        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
          URL: <code className="mono">{backendUrl}</code> · Status:{" "}
          <strong style={{ color: backendStatus === "connected" ? "var(--success)" : backendStatus === "failed" ? "var(--danger)" : "inherit" }}>
            {backendStatusText}
          </strong>
        </p>
        {backendDevices && (
          <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
            Recommended: {backendDevices.recommended.toUpperCase()} · CUDA:{" "}
            {backendDevices.available.cuda ? "on" : "off"} · MPS:{" "}
            {backendDevices.available.mps ? "on" : "off"} · CPU:{" "}
            {backendDevices.available.cpu ? "on" : "off"}
          </p>
        )}
        <div className="toolbar" style={{ alignItems: "flex-start" }}>
          <input
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            style={{ maxWidth: "540px", flex: 1 }}
            placeholder="/api or https://....ngrok-free.app/api"
          />
          <button
            className="btn"
            type="button"
            onClick={() => runBackendCheck(backendUrl)}
            disabled={backendStatus === "checking"}
          >
            Save & test
          </button>
          <button
            className="btn"
            type="button"
            onClick={runSystemCheck}
            disabled={backendStatus !== "connected" || backendDevicesStatus === "checking"}
          >
            Check system
          </button>
        </div>
        <div style={{ fontSize: "0.8rem", marginTop: "0.5rem", color: "var(--muted)" }}>
          {backendDevicesStatus === "checking" ? "Checking devices..." : backendDevicesStatus === "done" ? "System check done." : ""}
        </div>
      </div>
      <p className="mono" style={{ color: "var(--muted)" }}>
        {projectId}
      </p>
      {devices && (
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>{devices}</p>
      )}

      {err && (
        <div className="card" style={{ borderColor: "var(--danger)" }}>
          {err}
        </div>
      )}

      <div id="workspace-train" className="card">
        <h3 style={{ marginTop: 0 }}>2 · Train relevance (BioBERT)</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Paste JSON array: <code className="mono">pmid</code>,{" "}
          <code className="mono">text</code> (abstract or body),{" "}
          <code className="mono">label</code> (0 or 1). Optional{" "}
          <code className="mono">title</code> enables title+abstract pair encoding (DKDM-style).
          Imports deduplicate by PMID (first row kept).
        </p>
        <textarea
          value={articlesJson}
          onChange={(e) => setArticlesJson(e.target.value)}
          style={{ minHeight: "160px", fontFamily: "JetBrains Mono, monospace" }}
        />
        <input
          ref={trainFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onTrainFile}
        />
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => trainFileRef.current?.click()}
          >
            Import articles (CSV / Excel / PKL)
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() =>
              api
                .downloadTemplate(projectId, "articles")
                .catch((err) => setErr((err as Error).message))
            }
          >
            Download template (CSV)
          </button>
        </div>

        <div className="grid2" style={{ marginTop: "1rem" }}>
          <div>
            <label>Processor</label>
            <select
              value={trainCfg.processor}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  processor: e.target.value as Processor,
                })
              }
            >
              <option value="auto">auto (CUDA → MPS → CPU)</option>
              <option value="cuda">CUDA (NVIDIA GPU)</option>
              <option value="mps">MPS (Apple Metal)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div>
            <label>Base model</label>
            <input
              value={trainCfg.base_model}
              onChange={(e) =>
                setTrainCfg({ ...trainCfg, base_model: e.target.value })
              }
            />
          </div>
          <div>
            <label>Learning rate</label>
            <input
              type="number"
              step="any"
              value={trainCfg.learning_rate}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  learning_rate: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div>
            <label>Epochs</label>
            <input
              type="number"
              value={trainCfg.num_train_epochs}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  num_train_epochs: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Train batch size</label>
            <input
              type="number"
              value={trainCfg.per_device_train_batch_size}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  per_device_train_batch_size: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Eval batch size</label>
            <input
              type="number"
              value={trainCfg.per_device_eval_batch_size}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  per_device_eval_batch_size: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Weight decay</label>
            <input
              type="number"
              step="any"
              value={trainCfg.weight_decay}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  weight_decay: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div>
            <label>Max sequence length</label>
            <input
              type="number"
              value={trainCfg.max_length}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  max_length: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>FP16 (null = auto: on for CUDA only)</label>
            <select
              value={
                trainCfg.fp16 === null ? "auto" : trainCfg.fp16 ? "on" : "off"
              }
              onChange={(e) => {
                const v = e.target.value;
                setTrainCfg({
                  ...trainCfg,
                  fp16: v === "auto" ? null : v === "on",
                });
              }}
            >
              <option value="auto">auto</option>
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </div>
          <div>
            <label>Validation fraction (single train)</label>
            <input
              type="number"
              step="0.05"
              min={0.1}
              max={0.4}
              value={valSplit}
              onChange={(e) => setValSplit(parseFloat(e.target.value))}
            />
          </div>
          <div>
            <label>K-fold splits (k-fold only)</label>
            <input
              type="number"
              min={2}
              max={10}
              value={kfoldSplits}
              onChange={(e) => setKfoldSplits(parseInt(e.target.value, 10))}
            />
          </div>
        </div>

        <div className="toolbar" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="btn"
            disabled={!isBackendConnected || busy}
            onClick={applyTrainPreset}
          >
            Apply MPS safe preset
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || busy}
            onClick={() => train(false)}
          >
            Start training (holdout)
          </button>
          <button
            type="button"
            className="btn"
            disabled={!isBackendConnected || busy}
            onClick={() => train(true)}
          >
            Start k-fold CV
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("integrations")}
          >
            Next: PubMed & LitSuggest
          </button>
        </div>

        {jobId && (
          <div style={{ marginTop: "1rem" }}>
            <span className="tag">job {jobId}</span>
            <pre
              className="mono"
              style={{
                marginTop: "0.5rem",
                padding: "0.75rem",
                background: "var(--bg)",
                borderRadius: 8,
                overflow: "auto",
                fontSize: "0.82rem",
              }}
            >
              {JSON.stringify(jobPoll, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <div id="workspace-integrations" className="card">
        <h3 style={{ marginTop: 0 }}>3 · PubMed (Entrez) fetch & LitSuggest comparison</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Fetch citations from NCBI (email required), then optionally compare your binary labels to LitSuggest
          scores (DKDM-style threshold on scores).
        </p>

        <h4 style={{ marginBottom: "0.5rem" }}>PubMed fetch</h4>
        <div className="grid2">
          <div>
            <label>Entrez email (required by NCBI)</label>
            <input
              value={pubmedEmail}
              onChange={(e) => setPubmedEmail(e.target.value)}
              placeholder="you@institution.edu"
              autoComplete="email"
              style={{ width: "100%" }}
            />
          </div>
          <div>
            <label>Max articles / PMIDs</label>
            <input
              type="number"
              min={1}
              max={10000}
              value={pubmedMax}
              onChange={(e) => setPubmedMax(Number(e.target.value) || 50)}
            />
          </div>
        </div>
        <label style={{ display: "block", marginTop: "0.75rem" }}>PubMed query (esearch)</label>
        <textarea
          value={pubmedQuery}
          onChange={(e) => setPubmedQuery(e.target.value)}
          style={{ minHeight: "72px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <div className="grid2" style={{ marginTop: "0.75rem" }}>
          <div>
            <label>Min abstract length (0 = off)</label>
            <input
              type="number"
              min={0}
              value={pubmedMinAbstract}
              onChange={(e) => setPubmedMinAbstract(Number(e.target.value) || 0)}
            />
            <p style={{ fontSize: "0.78rem", color: "var(--muted)", margin: "0.35rem 0 0" }}>
              Matches DKDM: skip very short abstracts when set (e.g. 200).
            </p>
          </div>
        </div>
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || pubmedBusy}
            onClick={() => void runPubmedFetch()}
          >
            {pubmedBusy ? "Fetching…" : "Fetch from PubMed"}
          </button>
          <button
            type="button"
            className="btn"
            disabled={!pubmedResult?.articles?.length}
            onClick={() => copyPubmedToPipelineJson()}
          >
            Send results → pipeline JSON
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("train")}
          >
            Back
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("pipeline")}
          >
            Next: pipeline
          </button>
        </div>
        {pubmedResult && (
          <pre
            className="mono"
            style={{
              marginTop: "0.75rem",
              padding: "0.75rem",
              background: "var(--bg)",
              borderRadius: 8,
              overflow: "auto",
              fontSize: "0.78rem",
              maxHeight: "240px",
            }}
          >
            {JSON.stringify(pubmedResult, null, 2)}
          </pre>
        )}

        <h4 style={{ marginTop: "1.5rem", marginBottom: "0.5rem" }}>LitSuggest comparison</h4>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Primary: your labels (e.g. from training export or <code className="mono">label</code> /{" "}
          <code className="mono">relevant</code>). Secondary: LitSuggest <code className="mono">pmid</code> +{" "}
          <code className="mono">score</code>. Import LitSuggest CSV or paste JSON.
        </p>
        <input
          ref={litsuggestFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl"
          style={{ display: "none" }}
          onChange={onLitSuggestFile}
        />
        <div className="toolbar" style={{ flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn"
            disabled={compareBusy}
            onClick={() => litsuggestFileRef.current?.click()}
          >
            Import LitSuggest file
          </button>
        </div>
        <label style={{ display: "block", marginTop: "0.75rem" }}>Primary labels (JSON array)</label>
        <textarea
          value={comparePrimaryJson}
          onChange={(e) => setComparePrimaryJson(e.target.value)}
          style={{ minHeight: "100px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <label style={{ display: "block", marginTop: "0.75rem" }}>LitSuggest scores (JSON array)</label>
        <textarea
          value={compareLitJson}
          onChange={(e) => setCompareLitJson(e.target.value)}
          style={{ minHeight: "100px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <div style={{ marginTop: "0.75rem" }}>
          <label>Score threshold for binary label (≥ = relevant)</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={compareThreshold}
            onChange={(e) => setCompareThreshold(Number(e.target.value))}
            style={{ width: "100%", maxWidth: "360px", verticalAlign: "middle" }}
          />{" "}
          <span className="mono">{compareThreshold.toFixed(2)}</span>
        </div>
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || compareBusy}
            onClick={() => void runLitCompare()}
          >
            {compareBusy ? "Comparing…" : "Run comparison"}
          </button>
        </div>
        {compareResult && (
          <pre
            className="mono"
            style={{
              marginTop: "0.75rem",
              padding: "0.75rem",
              background: "var(--bg)",
              borderRadius: 8,
              overflow: "auto",
              fontSize: "0.78rem",
              maxHeight: "320px",
            }}
          >
            {JSON.stringify(compareResult, null, 2)}
          </pre>
        )}
      </div>

      <div id="workspace-pipeline" className="card">
        <h3 style={{ marginTop: 0 }}>4 · Run pipeline</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Pick a trained <span className="mono">model_id</span>, choose a
          step or full workflow, then run on new abstracts (labels optional for
          inference).
        </p>
        <input
          ref={pipeFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onPipeArticlesFile}
        />
        <input
          ref={mentionFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onMentionsFile}
        />
        <div className="toolbar" style={{ marginTop: "0.5rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => pipeFileRef.current?.click()}
          >
            Import articles file
          </button>
          {mode === "normalize" && (
            <button
              type="button"
              className="btn"
              disabled={busy}
              onClick={() => mentionFileRef.current?.click()}
            >
              Import mentions file
            </button>
          )}
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() =>
              api
                .downloadTemplate(projectId, "mentions")
                .catch((err) => setErr((err as Error).message))
            }
          >
            Mentions template (CSV)
          </button>
        </div>
        <div className="grid2">
          <div>
            <label>Trained model</label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              <option value="">— select —</option>
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as PipelineMode)}
            >
              <option value="full">full (classify → NER → normalize)</option>
              <option value="classify">classify only</option>
              <option value="ner">NER only</option>
              <option value="normalize">normalize only</option>
            </select>
          </div>
          <div>
            <label>Processor (inference)</label>
            <select
              value={processor}
              onChange={(e) => setProcessor(e.target.value as Processor)}
            >
              <option value="auto">auto</option>
              <option value="cuda">cuda</option>
              <option value="mps">mps</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div>
            <label>Relevance threshold (classification)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={relevanceThreshold}
              onChange={(e) => setRelevanceThreshold(parseFloat(e.target.value))}
              disabled={mode === "ner" || mode === "normalize"}
            />
            <div className="tag" style={{ marginTop: "0.35rem" }}>
              {relevanceThreshold.toFixed(2)}
            </div>
          </div>
          <div>
            <label>NER model (Hugging Face id)</label>
            <input
              value={nerModel}
              onChange={(e) => setNerModel(e.target.value)}
            />
          </div>
          <div>
            <label>Batch size</label>
            <input
              type="number"
              min={1}
              max={64}
              value={pipeBatchSize}
              onChange={(e) => setPipeBatchSize(parseInt(e.target.value, 10))}
            />
          </div>
          <div>
            <label>Wikipedia fallback</label>
            <select
              value={pipeUseWikipedia ? "on" : "off"}
              onChange={(e) => setPipeUseWikipedia(e.target.value === "on")}
            >
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </div>
        </div>

        <div style={{ marginTop: "1rem" }}>
          <label>Articles JSON</label>
          <textarea
            value={pipeJson}
            onChange={(e) => setPipeJson(e.target.value)}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          />
        </div>

        {mode === "normalize" && (
          <div style={{ marginTop: "1rem" }}>
            <label>Mentions JSON (for normalize-only)</label>
            <textarea
              value={normJson}
              onChange={(e) => setNormJson(e.target.value)}
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            />
          </div>
        )}

        <div className="toolbar" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={applyPipelinePreset}
          >
            Apply MPS safe preset
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || busy || !modelId}
            onClick={runPipe}
          >
            Run pipeline
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("integrations")}
          >
            Back
          </button>
        </div>

        {lastRunInfo && lastRunInfo.files.length > 0 && (
          <div style={{ marginTop: "1rem" }}>
            <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
              Saved under <span className="mono">{lastRunInfo.path}</span>:{" "}
              {lastRunInfo.files.join(", ")}
            </p>
            <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
              Export last run (CSV needs only that step; Excel/PKL per table).
            </p>
            <div
              className="toolbar"
              style={{ marginTop: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}
            >
              {lastRunInfo.files.includes("classification.csv") && (
                <>
                  <span className="tag">classification</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "csv",
                          "classification.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "xlsx",
                          "classification.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "pkl",
                          "classification.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              {lastRunInfo.files.includes("mentions.csv") && (
                <>
                  <span className="tag">mentions</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "csv",
                          "mentions.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "xlsx",
                          "mentions.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "pkl",
                          "mentions.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              {lastRunInfo.files.includes("normalized.csv") && (
                <>
                  <span className="tag">normalized</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "csv",
                          "normalized.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "xlsx",
                          "normalized.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "pkl",
                          "normalized.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              <span className="tag">bundle</span>
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() =>
                  api
                    .downloadExport(
                      projectId,
                      "bundle",
                      "pkl",
                      "geneminer_last_run_bundle.pkl"
                    )
                    .catch((err) => setErr((err as Error).message))
                }
              >
                All tables PKL
              </button>
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() =>
                  api
                    .downloadExport(
                      projectId,
                      "bundle",
                      "xlsx",
                      "geneminer_last_run_bundle.xlsx"
                    )
                    .catch((err) => setErr((err as Error).message))
                }
              >
                All tables Excel
              </button>
            </div>
          </div>
        )}

        {pipeResult && (
          <div style={{ marginTop: "1rem" }}>
            <h4>Charts</h4>
            <ChartFromPayload pipeResult={pipeResult} />
            {filteredClassificationRows.length > 0 && (
              <div style={{ marginTop: "1rem" }}>
                <h4>
                  {`Classification rows filtered by probability (>= ${relevanceThreshold.toFixed(2)})`}
                </h4>
                <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
                  {filteredClassificationRows.length} / {classificationRows.length}{" "}
                  rows kept.
                </p>
                <pre
                  className="mono"
                  style={{
                    padding: "0.75rem",
                    background: "var(--bg)",
                    borderRadius: 8,
                    overflow: "auto",
                    maxHeight: "240px",
                    fontSize: "0.8rem",
                  }}
                >
                  {JSON.stringify(filteredClassificationRows, null, 2)}
                </pre>
              </div>
            )}
            <h4>Raw JSON</h4>
            <pre
              className="mono"
              style={{
                padding: "0.75rem",
                background: "var(--bg)",
                borderRadius: 8,
                overflow: "auto",
                maxHeight: "320px",
                fontSize: "0.8rem",
              }}
            >
              {JSON.stringify(pipeResult, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function ChartFromPayload({
  pipeResult,
}: {
  pipeResult: Record<string, unknown>;
}) {
  const charts = pipeResult.charts as Record<string, Record<string, number>> | undefined;
  if (!charts || typeof charts !== "object") {
    return <p style={{ color: "var(--muted)" }}>No chart data.</p>;
  }
  const entries = Object.entries(charts);
  if (entries.length === 0) {
    return null;
  }
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {entries.map(([title, obj]) => {
        const pairs = Object.entries(obj);
        const max = Math.max(1, ...pairs.map(([, v]) => v));
        return (
          <div key={title}>
            <div className="tag" style={{ marginBottom: "0.5rem" }}>
              {title}
            </div>
            <div className="bar-chart">
              {pairs.map(([k, v]) => (
                <div key={k} className="bar" style={{ height: `${(v / max) * 100}%` }}>
                  <span>{k}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
