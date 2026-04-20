"""PubMed Entrez fetch and LitSuggest-style label comparison."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.models import LitSuggestCompareRequest, PubMedFetchRequest
from app.services import dataset_io, project_service
from app.services.label_comparison import compare_pmids_labels_vs_scores
from app.services.pubmed_entrez import EntrezNotAvailableError

router = APIRouter(prefix="/projects", tags=["external-data"])

_MAX_BYTES = 50 * 1024 * 1024


async def _save_upload_temp(upload: UploadFile) -> Path:
    data = await upload.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, "File too large (max 50MB)")
    suffix = Path(upload.filename or "upload").suffix.lower() or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


def _run_pubmed_fetch(body: PubMedFetchRequest) -> Dict[str, Any]:
    from app.services import pubmed_entrez

    if "@" not in body.email:
        raise ValueError("NCBI Entrez requires a valid email address.")

    raw_pmids: List[str] = [
        str(p).strip() for p in (body.pmids or []) if str(p).strip()
    ]

    search_total: int | None = None
    if raw_pmids:
        pmid_list = raw_pmids[: body.max_results]
    else:
        if not body.query or not body.query.strip():
            raise ValueError("Provide a non-empty `pmids` list or a PubMed `query`.")
        pmid_list, total = pubmed_entrez.search_pubmed_ids(
            body.query.strip(),
            body.email,
            max_results=body.max_results,
            retstart=body.retstart,
        )
        search_total = total

    articles = pubmed_entrez.fetch_pubmed_articles(
        pmid_list,
        body.email,
        min_abstract_chars=body.min_abstract_chars,
        sleep_between_batches=body.sleep_between_batches,
    )

    return {
        "queried_id_count": len(pmid_list),
        "search_total_estimate": search_total,
        "row_count": len(articles),
        "articles": articles,
    }


@router.post("/{project_id}/data/pubmed/fetch")
def pubmed_fetch(project_id: str, body: PubMedFetchRequest) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    try:
        return _run_pubmed_fetch(body)
    except EntrezNotAvailableError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"PubMed / Entrez request failed: {e!s}") from e


@router.post("/{project_id}/data/import/litsuggest-scores")
async def import_litsuggest_scores(project_id: str, file: UploadFile = File(...)) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    path = await _save_upload_temp(file)
    try:
        _df, rows = dataset_io.read_litsuggest_scores_from_path(path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return {"kind": "litsuggest_scores", "row_count": len(rows), "litsuggest": rows}


@router.post("/{project_id}/data/compare/litsuggest")
def compare_litsuggest(project_id: str, body: LitSuggestCompareRequest) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if not body.primary:
        raise HTTPException(400, "`primary` rows must not be empty")
    if not body.litsuggest:
        raise HTTPException(400, "`litsuggest` rows must not be empty")
    try:
        return compare_pmids_labels_vs_scores(
            body.primary,
            body.litsuggest,
            score_threshold=body.score_threshold,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Comparison failed: {e!s}") from e
