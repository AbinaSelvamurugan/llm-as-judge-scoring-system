import json
from judgebench.schemas import JudgeConfig
from judgebench.judge import ClaudeJudge, prompt_snapshot, run_evaluations
from judgebench.experiments import paired_experiment, stability
from judgebench.reporting import report_dict, render_report
from test_judge import FakeClient, output


def test_repeat_and_robustness_with_mock(store, rubric):
    item = store.item("fixture-001")
    cfg = JudgeConfig(repeat_count=2, limit=1)
    judge = ClaudeJudge(cfg, client=FakeClient(output(item, rubric)))
    run_id = store.create_run(cfg, rubric, "synthetic-test-v1", prompt_snapshot("v1"))
    run_evaluations(store, run_id, judge)
    assert stability(store, run_id)["fixture-001"]["factual"]["unique_scores"] == 1
    report = paired_experiment(store, judge, "fixture-001", "presentation", "Four.", runs=2)
    assert report["kind"] == "presentation"
    assert store.experiments()[0]["type"] == "presentation"


def test_all_report_formats_no_humans(store, rubric):
    run = store.create_run(JudgeConfig(), rubric, "synthetic-test-v1", prompt_snapshot("v1"))
    payload = report_dict(store, run, n_boot=0)
    assert payload["sample"]["submitted_annotations"] == 0
    assert payload["agreement"]["criteria"]["factual"]["exact"] is None
    for fmt in ("json", "markdown", "html", "csv"):
        data = render_report(payload, fmt)
        assert data
        if fmt == "csv":
            assert b"eligible" in data
        if fmt == "json":
            assert json.loads(data)["run"]["run_id"] == run
