import json
import pytest
from pydantic import ValidationError
from judgebench.schemas import Rubric, Scale, CriterionJudgment
from judgebench.rubrics import validate_scores, aggregate
from judgebench.datasets import load_bytes, load_path, export_items


def test_rubric_active_anchors_and_weights(rubric):
    assert len(rubric.criteria) == 7
    assert sum(c.enabled for c in rubric.criteria) == 5
    # model_copy bypasses validation; all external inputs use model_validate.
    raw = rubric.model_dump()
    raw["criteria"][0]["anchors"] = {1: "one"}
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)


def test_invalid_criteria(rubric):
    raw = rubric.model_dump()
    raw["criteria"][0]["critical_below"] = 7
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)
    raw = rubric.model_dump()
    for c in raw["criteria"]:
        c["weight"] = 0
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)
    raw = rubric.model_dump()
    raw["criteria"][1]["key"] = raw["criteria"][0]["key"]
    with pytest.raises(ValidationError):
        Rubric.model_validate(raw)


def test_aggregate_critical_abstain(rubric):
    scores = {c.key: 5 for c in rubric.criteria if c.enabled}
    scores["grounding"] = None
    scores["factual"] = 1
    a = aggregate(rubric, scores)
    assert a.critical_failure
    assert "grounding" in a.abstained_criteria
    assert a.decision == "fail"  # explicit critical-failure override survives abstentions
    with pytest.raises(ValueError):
        validate_scores(rubric, {**scores, "unknown": 5})
    with pytest.raises(ValueError):
        validate_scores(rubric, {**scores, "factual": 6})


def test_jsonl_csv_roundtrip():
    original = load_path("data/synthetic_fixture.jsonl")
    assert len(original) == 3
    for fmt in ("jsonl", "csv"):
        data = export_items(original, fmt)
        assert load_bytes(data, "." + fmt) == original
    with pytest.raises(ValueError, match="duplicate"):
        load_bytes(export_items(original[:1] * 2, "jsonl"), ".jsonl")
    with pytest.raises(ValueError, match="only"):
        load_bytes(b"x", ".exe")
    with pytest.raises(ValueError, match="8 MB"):
        load_bytes(b"a" * 8000001, ".jsonl")


def test_invalid_dataset_reports_row():
    item = load_path("data/synthetic_fixture.jsonl")[0].model_dump()
    item["question"] = ""
    with pytest.raises(ValueError, match="record 1"):
        load_bytes((json.dumps(item) + "\n").encode(), ".jsonl")


def test_empty_reference_and_binary(rubric):
    assert rubric.criteria[-1].scale == Scale.binary
    assert sorted(rubric.criteria[-1].anchors) == [0, 1]
    with pytest.raises(ValidationError):
        CriterionJudgment(criterion="factual", score=None, rationale="unclear")
    with pytest.raises(ValidationError):
        CriterionJudgment(criterion="factual", score=3, rationale="okay", abstention_reason="missing")
