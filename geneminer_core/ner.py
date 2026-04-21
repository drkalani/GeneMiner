"""Gene/protein NER helpers.

Current supported backends:
- transformers token-classification pipeline (default, production path)
- optional legacy bent annotator parsing
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from importlib import import_module
from typing import Any, Callable, List, Optional, Tuple

from typing import Literal

from geneminer_core.devices import device_for_pipeline, pipeline_device_index
from geneminer_core.schemas import MentionRecord

DEFAULT_NER_MODEL = "pruas/BENT-PubMedBERT-NER-Gene"
NerMethod = Literal["transformers", "bent"]


def load_ner_pipeline(
    model_name: str = DEFAULT_NER_MODEL,
    processor: Optional[str] = None,
):
    transformers_pipeline = import_module("transformers").pipeline
    device = device_for_pipeline(processor)
    dev_idx = pipeline_device_index(device)
    return transformers_pipeline(
        "token-classification",
        model=model_name,
        tokenizer=model_name,
        aggregation_strategy="simple",
        device=dev_idx,
    )


def extract_entities(
    pairs: List[Tuple[str, str]],
    ner_pipeline,
) -> List[MentionRecord]:
    """pairs: list of (pmid, text)."""
    records: List[MentionRecord] = []
    for pmid, text in pairs:
        if not text or not str(text).strip():
            continue
        try:
            entities = ner_pipeline(text)
        except Exception:
            continue
        for ent in entities:
            score = ent.get("score")
            records.append(
                MentionRecord(
                    pmid=str(pmid),
                    mention=str(ent.get("word", "")).strip(),
                    start=int(ent.get("start", 0)),
                    end=int(ent.get("end", 0)),
                    score=float(score) if score is not None else None,
                )
            )
    return records


def _parse_bent_ann_lines(lines: List[str], pmid: str) -> List[MentionRecord]:
    records: List[MentionRecord] = []
    for line in lines:
        if not line.startswith("T"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue

        span = parts[1]
        nums = [int(n) for n in re.findall(r"\d+", span)]
        start = nums[0] if nums else 0
        end = nums[1] if len(nums) > 1 else 0
        mention = parts[2].strip()
        records.append(
            MentionRecord(
                pmid=str(pmid),
                mention=mention,
                start=start,
                end=end,
                score=None,
            )
        )
    return records


def _collect_ann_files(out_dir: Path, expected_count: int) -> List[Path]:
    def _sort_key(path: Path) -> tuple[object, ...]:
        parts = re.findall(r"\d+|[A-Za-z]+", path.stem)
        keys: List[object] = []
        for part in parts:
            if part.isdigit():
                keys.append(int(part))
            else:
                keys.append(part.lower())
        if not keys:
            return (path.name,)
        return tuple(keys)

    files = sorted(out_dir.rglob("*.ann"), key=_sort_key)
    if not files:
        return []
    if len(files) >= expected_count:
        return files[:expected_count]
    return files


def _invoke_bent_annotate(annotate_fn: Callable[..., Any], texts: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: List[dict[str, Any]] = [
        {
            "input_text": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": str(out_dir),
        },
        {
            "input_text": texts,
            "types": {"gene": ""},
            "task": "gene",
            "recognize": True,
            "out_dir": str(out_dir),
        },
        {
            "texts": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": str(out_dir),
        },
        {
            "text": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": str(out_dir),
        },
        {
            "input_text": texts,
            "types": ["gene"],
            "recognize": True,
            "out_dir": str(out_dir),
        },
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            annotate_fn(**kwargs)
            return
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            raise RuntimeError(f"Bent annotation failed: {exc}") from exc

    try:
        annotate_fn(texts, out_dir=str(out_dir), recognize=True)
        return
    except Exception as exc:
        raise RuntimeError(
            "Bent annotation call signature did not match known variants."
        ) from last_error or exc


def extract_entities_with_bent(pairs: List[Tuple[str, str]]) -> List[MentionRecord]:
    non_empty = [(pmid, text) for pmid, text in pairs if text and str(text).strip()]
    if not non_empty:
        return []

    try:
        bt = import_module("bent.annotate")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bent method requires `bent` package. Install the latest supported version explicitly: "
            "`pip install bent==0.0.80` (Python 3.10.x, <=3.10.13). "
            "You can use `scripts/setup_bent_runtime.sh` for guided local setup."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import bent annotator: {exc}") from exc
    if not hasattr(bt, "annotate"):
        raise RuntimeError("Bent package does not expose annotate() entry point.")

    texts = [text for _, text in non_empty]
    records: List[MentionRecord] = []

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "output" / "ner"

        try:
            _invoke_bent_annotate(bt.annotate, texts, out_dir)
        except Exception as exc:
            raise RuntimeError(f"Bent annotation failed: {exc}") from exc

        ann_files = _collect_ann_files(out_dir, len(non_empty))
        if not ann_files:
            raise RuntimeError(f"Bent annotation produced no .ann files in {out_dir}.")
        for ann_file, (pmid, _) in zip(ann_files, non_empty):
            lines = ann_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            records.extend(_parse_bent_ann_lines(lines, pmid))

    return records


def extract_entities_with_method(
    pairs: List[Tuple[str, str]],
    ner_pipeline=None,
    method: NerMethod = "transformers",
) -> List[MentionRecord]:
    if method == "bent":
        return extract_entities_with_bent(pairs)
    return extract_entities(pairs, ner_pipeline=ner_pipeline)
