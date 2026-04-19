# GeneMiner Platform

End-to-end system for **disease-agnostic** literature mining: train a **binary relevance** classifier on your labeled abstracts (any condition), run **gene/protein NER**, and **normalize** symbols (MyGene + optional Wikipedia fallback). Backend: **FastAPI**. UI: **React (Vite)**.

## Features

- **Projects**: separate workspace per disease/study (`disease_key` is metadata; models are trained on *your* labels).
- **Training**: holdout validation or **stratified k-fold**; choose **CUDA**, **Apple Metal (MPS)**, or **CPU**; tune LR, epochs, batch sizes, sequence length, FP16 (auto on CUDA by default).
- **Pipeline**: run **classify only**, **NER only**, **normalize only**, or **full** workflow.
- **REST API**: OpenAPI at `/docs` when the server is running.

## Requirements

- Python **3.10+**
- Node.js **18+** (for the frontend)
- PyTorch with the backend you need ([pytorch.org](https://pytorch.org))

## Setup

```bash
cd /path/to/GeneMiner-DKD
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[api]"
```

### Backend

```bash
cd backend
export PYTHONPATH=.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from the repo root:

```bash
chmod +x scripts/run_backend.sh
./scripts/run_backend.sh
```

Data and trained models are stored under `data/projects/` (created automatically).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). API calls are proxied to `http://127.0.0.1:8000` under `/api/*`.

#### Frontend API base configuration

- For local development (default), keep `VITE_API_BASE` unset.
- For Colab public endpoint, set:

```bash
cd frontend
cat > .env.local <<'EOF'
VITE_API_BASE=https://<your-ngrok-domain>
EOF
npm run dev
```

- You can also run with explicit env variable at command line:

```bash
VITE_API_BASE=https://<your-ngrok-domain> npm run dev
```

### Colab backend template

Run this notebook on Google Colab to expose the backend using a free GPU:

- `colab_geneminer_backend.ipynb`
- It installs backend dependencies, starts FastAPI in background, and opens an ngrok tunnel.
- Replace `REPO_URL` in the notebook with your own repository URL.

## Core library

Importable Python package `geneminer_core` (installed with `pip install -e .`):

- `geneminer_core.devices`: `resolve_torch_device("cuda"|"mps"|"cpu"|"auto")`
- `geneminer_core.relevance`: train + predict BioBERT classifier
- `geneminer_core.ner`: Hugging Face NER pipeline (default BENT-PubMedBERT)
- `geneminer_core.normalization`: character rules + MyGene + optional Wikipedia
- `geneminer_core.pipeline`: orchestration helpers

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/devices` | CUDA/MPS availability |
| GET/POST | `/projects` | List/create projects |
| GET | `/projects/{id}/models` | List trained model folders |
| POST | `/train/{id}/relevance` | Start training job (background) |
| POST | `/train/{id}/relevance/kfold` | Start k-fold job |
| GET | `/train/jobs/{job_id}` | Poll job status + metrics |
| POST | `/pipeline/run` | Run classify / ner / normalize / full |

## Notes

- **NER training** is not fine-tuned in this stack; the default is a pretrained `pruas/BENT-PubMedBERT-NER-Gene` model. You can swap `ner_model` in API/UI.
- **Normalization** uses live **MyGene** queries; rate-limiting sleep is applied. For high volume, add local caching (future improvement).
- **Jobs** are stored in memory; use Redis/Queue for multi-worker production.

## License

Use under your team’s policy; add a `LICENSE` file if you distribute publicly.
