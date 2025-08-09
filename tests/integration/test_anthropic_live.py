"""Opt-in live API smoke test. Never invoked by ordinary offline test runs."""
import os
import pytest
from judgebench.schemas import JudgeConfig
from judgebench.rubrics import load_rubric
from judgebench.datasets import load_path
from judgebench.judge import ClaudeJudge


@pytest.mark.skipif(os.getenv("RUN_LIVE_ANTHROPIC") != "1" or not os.getenv("ANTHROPIC_API_KEY"),
                    reason="explicit RUN_LIVE_ANTHROPIC=1 and an API key required")
def test_real_provider_structured_output():
    judge = ClaudeJudge(JudgeConfig(limit=1, max_retries=1))
    item = load_path("data/synthetic_fixture.jsonl")[0]
    judgment = judge.evaluate(item, load_rubric("configs/rubric.yaml"))["judgment"]
    assert judgment.evaluation_id == item.evaluation_id
    assert judgment.rubric_version == item.rubric_key
    assert {x.criterion for x in judgment.criteria} == {
        "factual", "relevance", "completeness", "grounding", "instruction"}
