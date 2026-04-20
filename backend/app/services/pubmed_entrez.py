"""PubMed Entrez (NCBI E-utilities) fetch — requires biopython."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency: install with `pip install biopython`
try:
    from Bio import Entrez, Medline
except ImportError:  # pragma: no cover - exercised when dep missing
    Entrez = None  # type: ignore[misc, assignment]
    Medline = None  # type: ignore[misc, assignment]


class EntrezNotAvailableError(RuntimeError):
    """Raised when BioPython is not installed."""


def _require_bio() -> Tuple[Any, Any]:
    if Entrez is None or Medline is None:
        raise EntrezNotAvailableError(
            "Biopython is required for PubMed Entrez. Install with: pip install biopython"
        )
    return Entrez, Medline


def _join_medline_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if x is not None)
    return str(value).strip()


def search_pubmed_ids(
    query: str,
    email: str,
    *,
    max_results: int,
    retstart: int = 0,
) -> Tuple[List[str], int]:
    """Return (pmid_list, total_count_from_search) for a PubMed query."""
    Entrez, _ = _require_bio()
    Entrez.email = email
    max_results = max(1, min(max_results, 100_000))
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        retstart=retstart,
        sort="relevance",
    )
    try:
        record = Entrez.read(handle)
    finally:
        handle.close()
    id_list = [str(x) for x in record.get("IdList", [])]
    total = int(record.get("Count", len(id_list)))
    return id_list, total


def _record_to_article(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pmid_raw = rec.get("PMID")
    if isinstance(pmid_raw, list):
        pmid = str(pmid_raw[0]) if pmid_raw else ""
    else:
        pmid = str(pmid_raw or "").strip()
    if not pmid:
        return None
    title = _join_medline_field(rec.get("TI"))
    abstract = _join_medline_field(rec.get("AB"))
    if not title and not abstract:
        return {
            "pmid": pmid,
            "title": "",
            "text": "",
        }
    return {
        "pmid": pmid,
        "title": title or "",
        "text": abstract,
    }


def fetch_pubmed_articles(
    pmids: List[str],
    email: str,
    *,
    batch_size: int = 200,
    sleep_between_batches: float = 0.12,
    min_abstract_chars: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch MEDLINE records for PMIDs (batched efetch + Medline.parse).

    `min_abstract_chars`: if > 0, drop rows where abstract length is below this
    (DKDM notebook used ~200 to skip very short snippets).
    """
    Entrez, Medline = _require_bio()
    Entrez.email = email
    seen: set[str] = set()
    uniq = []
    for p in pmids:
        k = str(p).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    if not uniq:
        return []
    out: List[Dict[str, Any]] = []
    batch_size = max(1, min(batch_size, 400))
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i : i + batch_size]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="medline",
            retmode="text",
        )
        try:
            for rec in Medline.parse(handle):
                art = _record_to_article(rec)
                if art is None:
                    continue
                abst = art.get("text") or ""
                if min_abstract_chars > 0 and len(abst.strip()) < min_abstract_chars:
                    continue
                out.append(art)
        finally:
            handle.close()
        if i + batch_size < len(uniq) and sleep_between_batches > 0:
            time.sleep(sleep_between_batches)
    out.sort(key=lambda r: r["pmid"])
    return out
