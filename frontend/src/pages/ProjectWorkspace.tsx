import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  type Article,
  type PipelineMode,
  type Processor,
  type TrainingConfig,
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

const exampleArticlesJson = `[
  {"pmid":"10000001","text":"TGF-beta signaling in diabetic nephropathy and kidney fibrosis.","label":1},
  {"pmid":"10000002","text":"Weather patterns in coastal regions unrelated to nephrology.","label":0}
]`;

export function ProjectWorkspace() {
  const { id } = useParams<{ id: string }>();
  const projectId = id ?? "";

  const [models, setModels] = useState<string[]>([]);
  const [devices, setDevices] = useState<string | null>(null);
  const [trainCfg, setTrainCfg] = useState<TrainingConfig>(defaultTrainConfig);
  const [valSplit, setValSplit] = useState(0.2);
  const [kfoldSplits, setKfoldSplits] = useState(5);
  const [articlesJson, setArticlesJson] = useState(exampleArticlesJson);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobPoll, setJobPoll] = useState<unknown>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [modelId, setModelId] = useState("");
  const [mode, setMode] = useState<PipelineMode>("full");
  const [pipeJson, setPipeJson] = useState(exampleArticlesJson);
  const [processor, setProcessor] = useState<Processor>("auto");
  const [nerModel, setNerModel] = useState("pruas/BENT-PubMedBERT-NER-Gene");
  const [pipeResult, setPipeResult] = useState<Record<string, unknown> | null>(
    null
  );
  const [normJson, setNormJson] = useState(
    `[{"pmid":"1","mention":"TGFB1","start":0,"end":5}]`
  );

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
    return raw.map((a) => ({
      pmid: String(a.pmid),
      text: String(a.text),
      label:
        a.label === undefined || a.label === null ? undefined : Number(a.label),
    }));
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
        articles: articles.map((a) => ({
          pmid: String(a.pmid),
          text: String(a.text),
          label:
            a.label === undefined || a.label === null
              ? undefined
              : Number(a.label),
        })),
        mode,
        processor,
        ner_model: nerModel,
        batch_size: 8,
        use_wikipedia_fallback: true,
      };
      if (mode === "normalize") {
        payload.mentions_json = JSON.parse(normJson) as Record<
          string,
          unknown
        >[];
      }
      const out = await api.runPipeline(payload);
      setPipeResult(out);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <p>
        <Link to="/projects">← Projects</Link>
      </p>
      <h1>Project workspace</h1>
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

      <div className="card">
        <h3 style={{ marginTop: 0 }}>1 · Train relevance (BioBERT)</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Paste JSON array of objects:{" "}
          <code className="mono">pmid</code>, <code className="mono">text</code>
          , <code className="mono">label</code> (0 or 1). Use enough examples per
          class.
        </p>
        <textarea
          value={articlesJson}
          onChange={(e) => setArticlesJson(e.target.value)}
          style={{ minHeight: "160px", fontFamily: "JetBrains Mono, monospace" }}
        />

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
            className="btn btn-primary"
            disabled={busy}
            onClick={() => train(false)}
          >
            Start training (holdout)
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => train(true)}
          >
            Start k-fold CV
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

      <div className="card">
        <h3 style={{ marginTop: 0 }}>2 · Run pipeline</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Pick a trained <span className="mono">model_id</span>, choose a
          step or full workflow, then run on new abstracts (labels optional for
          inference).
        </p>
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
            <label>NER model (Hugging Face id)</label>
            <input
              value={nerModel}
              onChange={(e) => setNerModel(e.target.value)}
            />
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
            className="btn btn-primary"
            disabled={busy || !modelId}
            onClick={runPipe}
          >
            Run pipeline
          </button>
        </div>

        {pipeResult && (
          <div style={{ marginTop: "1rem" }}>
            <h4>Charts</h4>
            <ChartFromPayload pipeResult={pipeResult} />
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
