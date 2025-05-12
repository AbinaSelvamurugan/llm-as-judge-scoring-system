"""Strict immutable data contracts; absence of evidence is never numeric zero."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Scale(str, Enum):
    ordinal = "ordinal"
    binary = "binary"


class Criterion(StrictModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=3)
    description: str = Field(min_length=10)
    instructions: str = Field(min_length=10)
    scale: Scale = Scale.ordinal
    anchors: dict[int, str]
    positive_example: str = Field(min_length=3)
    negative_example: str = Field(min_length=3)
    evidence_requirement: Literal["none", "preferred", "required"] = "preferred"
    abstain_when: str = Field(min_length=10)
    allow_abstain: bool = True
    weight: float = Field(default=1, ge=0)
    critical_below: int | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def check_anchors(self):
        expected = {0, 1} if self.scale == Scale.binary else {1, 2, 3, 4, 5}
        if set(self.anchors) != expected or any(len(x.strip()) < 12 for x in self.anchors.values()):
            raise ValueError(f"{self.key}: anchor set must be {sorted(expected)}, each >=12 chars")
        if self.critical_below is not None and self.critical_below not in expected:
            raise ValueError("critical_below must be a valid score on this criterion")
        return self


class Rubric(StrictModel):
    rubric_id: str = Field(pattern=r"^[a-z][a-z0-9_\-]*$")
    version: str = Field(min_length=1)
    title: str
    criteria: list[Criterion] = Field(min_length=1)
    pass_threshold: float | None = Field(default=None, ge=0, le=1)
    fail_on_critical: bool = True

    @model_validator(mode="after")
    def unique_criteria(self):
        active = [c for c in self.criteria if c.enabled]
        if len(set(c.key for c in self.criteria)) != len(self.criteria) or not active:
            raise ValueError("criterion keys must be unique and at least one active")
        if sum(c.weight for c in active) <= 0:
            raise ValueError("active criterion weights must sum to a positive number")
        return self

    @property
    def key(self) -> str:
        return f"{self.rubric_id}@{self.version}"


class EvaluationItem(StrictModel):
    evaluation_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    reference_answer: str | None = None
    evidence: str | None = None
    category: str = "general"
    split: Literal["development", "holdout", "unspecified"] = "unspecified"
    difficulty: str | None = None
    dataset_version: str = Field(min_length=1)
    rubric_key: str = Field(min_length=3)
    provenance: str = "unspecified"
    generator_model: str | None = None


class CriterionJudgment(StrictModel):
    criterion: str
    score: StrictInt | None
    rationale: str = Field(min_length=3)
    answer_excerpt: str | None = None
    evidence_locator: str | None = None
    error_category: Literal["none", "factual_error", "unsupported_claim", "contradiction",
                            "irrelevant", "omission", "instruction", "safety", "other"] = "none"
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def check_abstention(self):
        if (self.score is None) != (self.abstention_reason is not None):
            raise ValueError("abstentions need a reason; scores must not have an abstention reason")
        return self


class ProviderJudgment(StrictModel):
    evaluation_id: str
    rubric_version: str
    criteria: list[CriterionJudgment]


class Aggregate(StrictModel):
    unweighted_normalized: float | None
    weighted_normalized: float | None
    critical_failure: bool
    critical_criteria: list[str]
    missing_criteria: list[str]
    abstained_criteria: list[str]
    decision: Literal["pass", "fail", "not_assessable", "not_defined"]


class HumanScores(StrictModel):
    scores: dict[str, StrictInt | None]
    comments: str = ""
    insufficient_evidence: bool = False


class JudgeConfig(StrictModel):
    model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=8)
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    reference_policy: Literal["as_available", "require_reference", "no_reference"] = "as_available"
    evidence_policy: Literal["as_available", "require_evidence", "no_evidence"] = "as_available"
    batch_size: int = Field(default=5, ge=1, le=100)
    repeat_count: int = Field(default=1, ge=1, le=20)
    limit: int = Field(default=20, ge=1, le=50000)
    verify_model: bool = True
    structured_output: bool = True
    fallback_parse_retries: int = Field(default=1, ge=0, le=2)
    # Provider model identifiers must be checked via models.retrieve before requests.
