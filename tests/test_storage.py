import pytest
from judgebench.schemas import HumanScores, JudgeConfig
from judgebench.judge import prompt_snapshot


def complete():
    return {"factual": 5, "relevance": 5, "completeness": 4, "grounding": None, "instruction": 5}


def test_immutable_version_and_unknown(store, rubric):
    store.add_rubric(rubric)
    modified = rubric.model_copy(update={"title": "changed"})
    with pytest.raises(ValueError, match="immutable"):
        store.add_rubric(modified)
    with pytest.raises(ValueError, match="unknown rubric"):
        store.rubric("missing@1")


def test_independent_annotation_revision_and_adjudication(store):
    assert store.summary()["annotated_items"] == 0
    a = store.annotate("fixture-001", "rater_A", HumanScores(scores={"factual": 5}), "draft")
    assert store.latest_annotations(annotator_id="rater_A")[0]["state"] == "draft"
    with pytest.raises(ValueError):
        store.annotate("fixture-001", "rater_A", HumanScores(scores={"factual": 5}), "submitted")
    store.annotate("fixture-001", "rater_A", HumanScores(scores=complete()), "submitted")
    store.annotate("fixture-001", "rater_B", HumanScores(scores={**complete(), "factual": 4}), "submitted")
    assert len(store.latest_annotations(evaluation_id="fixture-001", submitted_only=True)) == 2
    assert len(store.annotation_history("fixture-001", "rater_A")) == 2
    store.annotate("fixture-001", "rater_A", HumanScores(scores=complete(), comments="clarified"), "submitted")
    with pytest.raises(ValueError, match="only be changed"):
        store.annotate("fixture-001", "rater_A", HumanScores(scores={}), "draft")
    store.adjudicate("fixture-001", "senior1", HumanScores(scores=complete()), "Referenced the original question")
    assert len(store.adjudications()) == 1
    assert len(store.annotation_history("fixture-001", "rater_A")) == 3
    assert store.summary()["annotated_items"] == 1


def test_adjudication_requires_two(store):
    store.annotate("fixture-001", "rater_A", HumanScores(scores=complete()))
    with pytest.raises(ValueError, match="two"):
        store.adjudicate("fixture-001", "senior", HumanScores(scores=complete()), "reason")


def test_run_and_failure_recovery(store, rubric):
    run = store.create_run(JudgeConfig(), rubric, "synthetic-test-v1", prompt_snapshot("v1"))
    store.save_result(run, "fixture-001", 0, error_type="rate_limit", error_message="failed")
    assert store.results(run)[0]["state"] == "failed"
    assert store.summary()["api_failures"] == 1
    store.set_run_state(run, "partial")
    assert store.run(run)["status"] == "partial"
    assert store.run(run)["completed_at"]


def test_run_items_frozen_and_multi_rubric_labels(store, rubric):
    second = rubric.model_copy(update={"version": "2.0.0", "title": "new rubric"})
    store.add_rubric(second)
    cfg = JudgeConfig(limit=2)
    rid = store.create_run(cfg, second, "synthetic-test-v1", prompt_snapshot("v2"))
    assert len(store.run_items(rid)) == 2
    assert store.run(rid)["rubric_key"] == "default@2.0.0"
    store.annotate("fixture-001", "rater_A", HumanScores(scores=complete()), rubric_key="default@2.0.0")
    assert len(store.latest_annotations(evaluation_id="fixture-001", rubric_key="default@2.0.0")) == 1
    assert len(store.latest_annotations(evaluation_id="fixture-001", rubric_key="default@1.0.0")) == 0
