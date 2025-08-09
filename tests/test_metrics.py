import numpy as np
import pytest
from judgebench.schemas import Scale, HumanScores, JudgeConfig
from judgebench.metrics import agreement, bootstrap_ci, human_human, compare_run, human_reference
from judgebench.judge import prompt_snapshot


def test_perfect_agreement():
    pairs = [(str(i), x, x) for i,x in enumerate([1,2,3,4,5])]
    m = agreement(pairs, [1,2,3,4,5], Scale.ordinal, bootstrap_replicates=100)
    assert m["exact"] == 1
    assert m["mae"] == 0
    assert m["kappa"] == 1
    assert m["weighted_kappa"] == 1
    assert m["spearman"] == pytest.approx(1)
    assert sum(sum(x) for x in m["confusion_counts"]) == 5
    assert m["exact_ci"] == [1,1]


def test_human_axes_bias_and_missing():
    p = [("a", 1, 3), ("b", 4, 5), ("c", 5, 5), ("d", None, 2), ("e", 3, None)]
    m = agreement(p, [1,2,3,4,5], Scale.ordinal, bootstrap_replicates=0)
    assert m["eligible"] == 3
    assert m["total_pairs"] == 5
    assert m["missing_human_or_abstained"] == 1
    assert m["missing_judge_or_abstained"] == 1
    assert m["exact"] == pytest.approx(1/3)
    assert m["within_one"] == pytest.approx(2/3)
    assert m["mae"] == 1
    assert m["bias"] == 1
    assert m["confusion_counts"][0][2] == 1  # human=1, judge=3
    assert m["exact_ci"] is None


def test_binary_f1_precision_recall():
    m = agreement([("1",1,1), ("2",1,0), ("3",0,1), ("4",0,0)], [0,1], Scale.binary, bootstrap_replicates=0)
    assert m["binary"] == {"positive_label": 1, "precision": .5, "recall": .5, "f1": .5}
    assert m["within_one"] is None
    assert m["kappa"] == 0
    assert m["confusion_counts"] == [[1,1],[1,1]]


def test_undefined_metrics_empty():
    assert agreement([("a", None, 2)], [1,2,3,4,5], Scale.ordinal)["exact"] is None
    assert agreement([("a", 3, 3)], [1,2,3,4,5], Scale.ordinal)["kappa"] is None
    assert bootstrap_ci([], lambda h,j: float(np.mean(h==j))) is None
    with pytest.raises(ValueError):
        agreement([("a", 6, 4)], [1,2,3,4,5], Scale.ordinal)


def test_human_reference_and_human_human(store, rubric):
    full = {c.key: 4 for c in rubric.criteria if c.enabled}
    for name, score in [("u1", 5), ("u2", 4), ("u3", 4)]:
        store.annotate("fixture-001", name, HumanScores(scores={**full, "factual": score}))
    ref = human_reference(store, "majority")
    assert ref[("fixture-001", rubric.key)][0]["factual"] == 4
    assert human_reference(store, "adjudicated") == {}
    hh = human_human(store, rubric.key, bootstrap_replicates=0)
    assert hh["factual"]["eligible"] == 3  # three human pairs; one item
    assert hh["factual"]["exact"] == pytest.approx(1/3)


def test_no_real_pairs_empty(store, rubric):
    rid = store.create_run(JudgeConfig(), rubric, "synthetic-test-v1", prompt_snapshot("v1"))
    m = compare_run(store, rid, "adjudicated", bootstrap_replicates=0)
    assert all(c["eligible"] == 0 for c in m["criteria"].values())


def test_agreement_category_filter_and_holdout_guard(store, rubric):
    from judgebench.metrics import compare_runs_holdout
    first = store.create_run(JudgeConfig(), rubric, "synthetic-test-v1", prompt_snapshot("v1"))
    second = store.create_run(JudgeConfig(), rubric, "synthetic-test-v1", prompt_snapshot("v2"))
    filtered = compare_run(store, first, "single", category_filter=["geography"], human_score_range=(4,5))
    assert filtered["criteria"]["factual"]["eligible"] == 0
    with pytest.raises(ValueError, match="holdout"):
        compare_runs_holdout(store, first, second)
