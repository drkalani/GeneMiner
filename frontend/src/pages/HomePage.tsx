import { Link } from "react-router-dom";

export function HomePage() {
  return (
    <div style={{ maxWidth: "720px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>
        Literature mining for any disease domain
      </h1>
      <p style={{ color: "var(--muted)", fontSize: "1.05rem" }}>
        GeneMiner helps researchers train a relevance classifier on{" "}
        <strong>your</strong> labeled abstracts, run gene/protein NER, and
        normalize symbols—on GPU, Apple Silicon Metal, or CPU.
      </p>

      <div className="steps">
        <span className="step-pill active">1 · Train</span>
        <span className="step-pill active">2 · Extract</span>
        <span className="step-pill active">3 · Normalize</span>
        <span className="step-pill active">4 · Review</span>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Designed for researchers</h3>
        <ul style={{ margin: 0, paddingLeft: "1.2rem", color: "var(--muted)" }}>
          <li>Create a project per disease or study (not only DKD).</li>
          <li>Upload labeled abstracts (0 = not relevant, 1 = relevant).</li>
          <li>Choose processor and full fine-tuning parameters.</li>
          <li>Run each pipeline step alone, or the full workflow end-to-end.</li>
        </ul>
      </div>

      <p>
        Open <Link to="/projects">Projects</Link> to begin.
      </p>
    </div>
  );
}
