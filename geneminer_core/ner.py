"""Gene/protein NER using BENT-PubMedBERT."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from transformers import pipeline

from geneminer_core.devices import device_for_pipeline, pipeline_device_index
from geneminer_core.schemas import MentionRecord

DEFAULT_NER_MODEL = "pruas/BENT-PubMedBERT-NER-Gene"


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
