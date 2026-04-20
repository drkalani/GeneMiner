# GeneMiner Platform

End-to-end system for **disease-agnostic** literature mining: train a **binary relevance** classifier on your labeled abstracts (any condition), run **gene/protein NER**, and **normalize** symbols (MyGene + optional Wikipedia fallback). Backend: **FastAPI**. UI: **React (Vite)**.

## Features

- **Projects**: separate workspace per disease/study (`disease_key` is metadata; models are trained on *your* labels).
- **Training**: holdout validation or **stratified k-fold**; choose **CUDA**, **Apple Metal (MPS)**, or **CPU**; tune LR, epochs, batch sizes, sequence length, FP16 (auto on CUDA by default).
- **Pipeline**: run **classify only**, **NER only**, **normalize only**, or **full** workflow.
- **Datasets**: import articles or mentions from **CSV / Excel / pickle**; export each step or a **bundle** (multi-sheet Excel or dict of DataFrames in PKL); CSV templates from the UI and API.
- **REST API**: OpenAPI at `/docs` when the server is running.

Pipeline runs write CSV snapshots under `data/projects/<id>/outputs/last_run/` (classification, mentions, normalized) so exports always have a stable on-disk source.

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

You can also run a one-command smoke check (works both local and Colab) after backend is up:

```bash
python scripts/colab_smoke_check.py --base-url "https://<your-ngrok-domain>"
```

For PubMed fetch check include email (recommended) and keep `max_results` small:

```bash
python scripts/colab_smoke_check.py \
  --base-url "https://<your-ngrok-domain>" \
  --email your_email@example.edu
```

If your Colab runtime blocks external NCBI requests, skip this step:

```bash
python scripts/colab_smoke_check.py \
  --base-url "https://<your-ngrok-domain>" \
  --skip-pubmed
```

### Colab notebook smoke snippet

You can paste this directly in Colab as a cell (after `BASE_URL` is set):

```python
import os
import requests

base = BASE_URL.rstrip("/")
probe_email = "you@example.edu"  # optional when skipping pubmed

print("health", requests.get(f"{base}/health", timeout=20).status_code)
pr = requests.post(f"{base}/projects", json={"name":"Colab smoke","disease_key":"colab","description":"smoke"}, timeout=20)
print("create project", pr.status_code, pr.text[:200])
pid = pr.json()["id"]

cmp = requests.post(
    f"{base}/projects/{pid}/data/compare/litsuggest",
    json={
        "primary":[{"pmid":"10","label":1},{"pmid":"11","label":0}],
        "litsuggest":[{"pmid":"10","score":0.87},{"pmid":"11","score":0.15}],
        "score_threshold":0.5,
    },
    timeout=20,
)
print("compare", cmp.status_code, cmp.json())

fetch = requests.post(
    f"{base}/projects/{pid}/data/pubmed/fetch",
    json={"email":probe_email, "query": '\"diabetic kidney disease\"[Title/Abstract]', "max_results":5},
    timeout=60,
)
print("pubmed fetch", fetch.status_code)
print(fetch.json() if fetch.headers.get("content-type","").startswith("application/json") else fetch.text[:200])
```

### Docker deployment (recommended)

Build and run both backend and frontend in containers:

```bash
docker compose up --build
```

Open:

- App UI: [http://localhost](http://localhost)  (Nginx proxy on port 80)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

This compose stack:

- Starts `backend` (FastAPI) on port `8000`.
- Builds and serves `frontend` with Nginx.
- Proxies API calls from the UI at `/api/*` to `backend:8000`.

You can find deployment files here:

- `backend/Dockerfile`
- `frontend/Dockerfile`
- `frontend/nginx.conf`
- `docker-compose.yml`
- `.dockerignore`

If you want a custom API base at build time:

```bash
docker compose build --build-arg VITE_API_BASE=https://api.example.com frontend
docker compose up frontend
```

Project artifacts are persisted with `./data:/app/data` in `docker-compose.yml`.

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
| GET | `/train/jobs` | List recent jobs (optional `project_id` filter) |
| GET | `/train/jobs` | List recent jobs (optional `project_id`, `limit`) |
| GET | `/train/jobs/{job_id}` | Poll job status + metrics |
| POST | `/pipeline/run` | Run classify / ner / normalize / full |
| GET | `/projects/{id}/data/last-run` | List files in `outputs/last_run` |
| POST | `/projects/{id}/data/import/articles` | `multipart/form-data` file → JSON articles |
| POST | `/projects/{id}/data/import/mentions` | File → JSON mention rows (normalize step) |
| GET | `/projects/{id}/data/export/{artifact}?format=` | `artifact` = `classification`, `mentions`, `normalized` (`csv` / `xlsx` / `pkl`) |
| GET | `/projects/{id}/data/export/bundle?format=` | `pkl` or `xlsx` (all tables present on disk) |
| GET | `/projects/{id}/data/templates/articles` | Download CSV column template |
| GET | `/projects/{id}/data/templates/mentions` | Mentions CSV template |
| POST | `/projects/{id}/data/pubmed/fetch` | Fetch abstracts from PubMed via Entrez |
| POST | `/projects/{id}/data/import/litsuggest-scores` | Upload LitSuggest score rows (`pmid`,`score`) |
| POST | `/projects/{id}/data/compare/litsuggest` | Compare model labels against LitSuggest scores |

## Notes

- **NER training** is not fine-tuned in this stack; the default is a pretrained `pruas/BENT-PubMedBERT-NER-Gene` model. You can swap `ner_model` in API/UI.
- **Normalization** uses live **MyGene** queries; rate-limiting sleep is applied. For high volume, add local caching (future improvement).
- **Jobs** are stored in memory; use Redis/Queue for multi-worker production.

## License

Use under your team’s policy; add a `LICENSE` file if you distribute publicly.
