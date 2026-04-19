"""Pydantic API models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ProcessorType = Literal["auto", "cuda", "mps", "cpu"]
PipelineMode = Literal["classify", "ner", "normalize", "full"]


class ArticleInput(BaseModel):
    pmid: str
    text: str
    label: Optional[int] = Field(None, description="0 irrelevant, 1 relevant (for training/eval)")


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    disease_key: str = Field(
        "custom",
        description="Identifier for disease/domain (e.g. dkd, alzheimer). Used for organization only.",
    )
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    disease_key: str
    description: str


class TrainingConfig(BaseModel):
    processor: ProcessorType = "auto"
    base_model: str = "dmis-lab/biobert-v1.1"
    learning_rate: float = 2e-5
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42
    max_length: int = 512
    fp16: Optional[bool] = None


class KFoldTrainingConfig(TrainingConfig):
    n_splits: int = Field(5, ge=2, le=10)


class TrainJobCreate(BaseModel):
    articles: List[ArticleInput]
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    validation_split: float = Field(0.2, ge=0.1, le=0.4)


class KFoldTrainJobCreate(BaseModel):
    articles: List[ArticleInput]
    config: KFoldTrainingConfig = Field(default_factory=KFoldTrainingConfig)


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    result: Optional[Dict[str, Any]] = None


class PipelineRunRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., description="Trained relevance model folder name under project")
    articles: List[ArticleInput]
    mode: PipelineMode = "full"
    processor: ProcessorType = "auto"
    ner_model: str = "pruas/BENT-PubMedBERT-NER-Gene"
    batch_size: int = 16
    use_wikipedia_fallback: bool = True
    # For normalize-only: pass prior mentions as rows
    mentions_json: Optional[List[Dict[str, Any]]] = None


class DeviceInfo(BaseModel):
    available: Dict[str, bool]
    recommended: str
