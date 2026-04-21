"""Gene/protein NER helpers.

Current supported backends:
- transformers token-classification pipeline (default, production path)
- optional legacy bent annotator parsing
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from typing import Literal

from transformers import pipeline

from geneminer_core.devices import device_for_pipeline, pipeline_device_index
from geneminer_core.schemas import MentionRecord

DEFAULT_NER_MODEL = "pruas/BENT-PubMedBERT-NER-Gene"
NerMethod = Literal["transformers", "bent"]


def load_ner_pipeline(
    model_name: str = DEFAULT_NER_MODEL,
    processor: Optional[str] = None,
):
    device = device_for_pipeline(processor)
    dev_idx = pipeline_device_index(device)
    return pipeline(
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


def extract_entities_with_bent(pairs: List[Tuple[str, str]]) -> List[MentionRecord]:
    non_empty = [(pmid, text) for pmid, text in pairs if text and str(text).strip()]
    if not non_empty:
        return []

    try:
        import bent.annotate as bt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bent method requires `bent` package. Install with `pip install bent`."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import bent annotator: {exc}") from exc

    texts = [text for _, text in non_empty]
    records: List[MentionRecord] = []

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "output" / "ner"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            bt.annotate(
                recognize=True,
                types={"gene": ""},
                input_text=texts,
                out_dir=str(out_dir),
            )
        except Exception as exc:
            raise RuntimeError(f"Bent annotation failed: {exc}") from exc

        for idx, (pmid, _) in enumerate(non_empty, start=1):
            candidates = [out_dir / f"doc_{idx}.ann", out_dir / f"doc_{idx-1}.ann"]
            ann_file = next((path for path in candidates if path.exists()), None)
            if ann_file is None:
                continue

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
