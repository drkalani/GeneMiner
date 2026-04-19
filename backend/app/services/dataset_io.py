"""Import/export tabular datasets (CSV, Excel, pickle) for articles and pipeline outputs."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from app.services.project_service import project_dir

ArtifactKind = Literal["classification", "mentions", "normalized", "bundle"]
ExportFormat = Literal["csv", "xlsx", "pkl"]

_ARTICLE_TEXT = ("text", "abstract", "title_abstract", "body", "content")
_ARTICLE_PMID = ("pmid", "pm_id", "id", "article_id", "pubmed", "PMID")
_ARTICLE_LABEL = ("label", "relevant", "y", "class", "target")

_MENTION_PMID = ("pmid", "PMID", "id")
_MENTION_TEXT = ("mention", "entity", "gene", "text", "word")


def _first_matching_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def read_articles_from_path(path: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load articles from .csv, .xlsx/.xls, or .pkl (DataFrame or list[dict])."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".pkl":
        obj = pd.read_pickle(path)  # noqa: S301 — trusted project uploads only
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict) and "articles" in obj:
            inner = obj["articles"]
            df = inner if isinstance(inner, pd.DataFrame) else pd.DataFrame(inner)
        else:
            raise ValueError("Pickle must contain DataFrame, list of rows, or dict with 'articles' key")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .pkl")

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("File has no rows")

    col_pmid = _first_matching_column(df, _ARTICLE_PMID)
    col_text = _first_matching_column(df, _ARTICLE_TEXT)
    if not col_pmid or not col_text:
        raise ValueError(
            "Articles file needs identifiable columns for PMID/id and text/abstract "
            f"(found columns: {list(df.columns)})"
        )
    col_label = _first_matching_column(df, _ARTICLE_LABEL)

    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pmid = str(row[col_pmid]).strip()
        text = str(row[col_text]).strip()
        if not pmid or not text:
            continue
        item: Dict[str, Any] = {"pmid": pmid, "text": text}
        if col_label is not None and pd.notna(row[col_label]):
            try:
                item["label"] = int(row[col_label])
            except (TypeError, ValueError):
                item["label"] = None
        out_rows.append(item)

    if not out_rows:
        raise ValueError("No valid article rows after parsing")

    slim = pd.DataFrame(out_rows)
    return slim, out_rows


def read_mentions_from_path(path: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load mention rows for normalize step."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".pkl":
        obj = pd.read_pickle(path)  # noqa: S301
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("File has no rows")

    col_pmid = _first_matching_column(df, _MENTION_PMID)
    col_mention = _first_matching_column(df, _MENTION_TEXT)
    if not col_pmid or not col_mention:
        raise ValueError(
            "Mentions file needs pmid and mention/entity columns "
            f"(found: {list(df.columns)})"
        )

    start_col = _first_matching_column(df, ("start", "start_offset", "begin"))
    end_col = _first_matching_column(df, ("end", "end_offset", "stop"))
    score_col = _first_matching_column(df, ("score", "confidence", "prob"))

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {
            "pmid": str(row[col_pmid]).strip(),
            "mention": str(row[col_mention]).strip(),
        }
        if start_col and pd.notna(row.get(start_col)):
            rec["start"] = int(row[start_col])
        else:
            rec["start"] = 0
        if end_col and pd.notna(row.get(end_col)):
            rec["end"] = int(row[end_col])
        else:
            rec["end"] = len(rec["mention"])
        if score_col and pd.notna(row.get(score_col)):
            try:
                rec["score"] = float(row[score_col])
            except (TypeError, ValueError):
                pass
        rows.append(rec)

    return pd.DataFrame(rows), rows


def last_run_dir(project_id: str) -> Path:
    return project_dir(project_id) / "outputs" / "last_run"


def artifact_path(project_id: str, name: str) -> Path:
    """name: classification | mentions | normalized (without extension)."""
    return last_run_dir(project_id) / f"{name}.csv"


def load_artifact_dataframe(project_id: str, name: str) -> pd.DataFrame:
    p = artifact_path(project_id, name)
    if not p.exists():
        raise FileNotFoundError(f"No saved {name} for this project (run pipeline first)")
    return pd.read_csv(p)


def export_dataframe_bytes(df: pd.DataFrame, fmt: ExportFormat, sheet_name: str = "data") -> Tuple[bytes, str]:
    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8"), "text/csv"
    if fmt == "xlsx":
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        bio.seek(0)
        return bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if fmt == "pkl":
        return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL), "application/octet-stream"
    raise ValueError(f"Unknown format {fmt}")


def export_bundle_bytes(project_id: str, fmt: Literal["pkl", "xlsx"]) -> Tuple[bytes, str, str]:
    names = ("classification", "mentions", "normalized")
    frames: Dict[str, pd.DataFrame] = {}
    for n in names:
        p = artifact_path(project_id, n)
        if p.exists():
            frames[n] = pd.read_csv(p)

    if not frames:
        raise FileNotFoundError("No pipeline outputs found; run the pipeline at least once")

    if fmt == "pkl":
        return (
            pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL),
            "application/octet-stream",
            "geneminer_last_run_bundle.pkl",
        )

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, fdf in frames.items():
            fdf.to_excel(writer, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return (
        bio.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "geneminer_last_run_bundle.xlsx",
    )


def articles_template_csv_bytes() -> bytes:
    df = pd.DataFrame(
        [
            {"pmid": "10000001", "text": "Example abstract text.", "label": 1},
            {"pmid": "10000002", "text": "Unrelated example text.", "label": 0},
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def mentions_template_csv_bytes() -> bytes:
    df = pd.DataFrame(
        [
            {"pmid": "10000001", "mention": "TGFB1", "start": 10, "end": 15, "score": 0.99},
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
