"""Item-level, criterion-aware agreement, missingness and bootstrap uncertainty."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, precision_recall_fscore_support)
from .schemas import Scale


def bootstrap_ci(pairs: list[tuple[str, float, float]], metric, n_boot: int = 500,
                 seed: int = 42, level: float = .95):
    """Cluster bootstrap by evaluation_id, preserving within-item correlated ratings."""
    unique = sorted({id_ for id_, _, _ in pairs})
    if len(unique) < 2 or n_boot < 2:
        return None
    groups = {id_: [(x, y) for k, x, y in pairs if k == id_] for id_ in unique}
    gen = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        ids = gen.choice(unique, size=len(unique), replace=True)
        observations = [p for id_ in ids for p in groups[id_]]
        try:
            result = metric(np.array([x for x, _ in observations]),
                            np.array([y for _, y in observations]))
            if result is not None and np.isfinite(result):
                samples.append(float(result))
        except (ValueError, ZeroDivisionError):
            pass
    if len(samples) < 2:
        return None
    alpha = (1 - level) / 2
    return [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))]


def agreement(pairs: list[tuple[str, int | float | None, int | float | None]],
              labels: list[int], scale: Scale, within_distance: int = 1,
              bootstrap_replicates: int = 500, seed: int = 42) -> dict:
    """Human label is second element; predicted judge label is third."""
    total = len(pairs)
    valid = [(id_, float(h), float(j)) for id_, h, j in pairs if h is not None and j is not None]
    human_missing = sum(h is None for _, h, _ in pairs)
    judge_missing = sum(j is None for _, _, j in pairs)
    result = {"total_pairs": total, "eligible": len(valid),
              "missing_human_or_abstained": human_missing,
              "missing_judge_or_abstained": judge_missing,
              "missing_policy": "pairwise complete-case; null/abstention excluded from point metrics",
              "labels": labels, "scale": scale.value,
              "human_distribution": {}, "judge_distribution": {},
              "confusion_counts": None, "exact": None, "exact_ci": None,
              "within_one": None, "within_one_ci": None, "mae": None,
              "mae_ci": None, "bias": None, "kappa": None, "weighted_kappa": None,
              "spearman": None, "kendall": None, "binary": None,
              "score_differences": []}
    if not valid:
        return result
    human = np.array([h for _, h, _ in valid]); judge = np.array([j for _, _, j in valid])
    if not np.all(np.isin(human, labels)) or not np.all(np.isin(judge, labels)):
        raise ValueError("scores outside criterion's label set; mean references require separate numeric analysis")
    score_diffs = judge - human
    result.update(human_distribution={str(k): int(v) for k, v in sorted(Counter(human.astype(int)).items())},
                  judge_distribution={str(k): int(v) for k, v in sorted(Counter(judge.astype(int)).items())},
                  confusion_counts=confusion_matrix(human, judge, labels=labels).tolist(),
                  exact=float(np.mean(human == judge)),
                  mae=float(np.mean(abs(score_diffs))), bias=float(np.mean(score_diffs)),
                  score_differences=score_diffs.tolist())
    result["exact_ci"] = bootstrap_ci(valid, lambda h, j: np.mean(h == j), bootstrap_replicates, seed)
    result["mae_ci"] = bootstrap_ci(valid, lambda h, j: np.mean(abs(j - h)), bootstrap_replicates, seed)
    if scale == Scale.ordinal:
        result["within_one"] = float(np.mean(abs(score_diffs) <= within_distance))
        result["within_one_ci"] = bootstrap_ci(valid, lambda h, j: np.mean(abs(j-h) <= within_distance),
                                               bootstrap_replicates, seed)
    # Undefined for constant-label distributions; suppress sklearn warnings for degenerate kappa.
    if len(np.unique(np.concatenate([human, judge]))) > 1:
        result["kappa"] = float(cohen_kappa_score(human, judge, labels=labels))
        if scale == Scale.ordinal:
            result["weighted_kappa"] = float(cohen_kappa_score(human, judge, labels=labels, weights="quadratic"))
    if scale == Scale.ordinal and len(np.unique(human)) > 1 and len(np.unique(judge)) > 1:
        result["spearman"] = float(spearmanr(human, judge).statistic)
        result["kendall"] = float(kendalltau(human, judge).statistic)
    if scale == Scale.binary:
        p, r, f, _ = precision_recall_fscore_support(human.astype(int), judge.astype(int),
                                                      labels=[1], zero_division=0)
        result["binary"] = {"positive_label": 1, "precision": float(p[0]),
                            "recall": float(r[0]), "f1": float(f[0])}
    return result


def _human_scores(row):
    return json.loads(row["scores"])["scores"]


def human_reference(store, strategy: str = "adjudicated") -> dict:
    """Return explicit per-item human reference and its construction provenance."""
    annotations = defaultdict(list)
    for a in store.latest_annotations(submitted_only=True):
        annotations[(a["evaluation_id"], a["rubric_key"])].append(a)
    adjudications = defaultdict(list)
    for a in store.adjudications():
        adjudications[(a["evaluation_id"], a["rubric_key"])].append(a)
    result = {}
    for key, rows in annotations.items():
        if strategy == "adjudicated":
            if not adjudications[key]:
                continue
            a = adjudications[key][-1]
            result[key] = (_human_scores(a), "adjudicated")
        elif strategy == "single":
            result[key] = (_human_scores(sorted(rows, key=lambda x: x["annotator_id"])[0]),
                           "single:first_annotator_id")
        elif strategy == "majority":
            scores = {}
            for c in store.rubric(key[1]).criteria:
                if not c.enabled:
                    continue
                vals = [v for a in rows if (v := _human_scores(a).get(c.key)) is not None]
                counts = Counter(vals)
                if counts:
                    top = counts.most_common()
                    scores[c.key] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
                else:
                    scores[c.key] = None
            result[key] = (scores, "majority:ties_abstained")
        elif strategy == "mean":
            scores = {}
            for c in store.rubric(key[1]).criteria:
                if c.enabled:
                    vals = [v for a in rows if (v := _human_scores(a).get(c.key)) is not None]
                    scores[c.key] = float(np.mean(vals)) if vals else None
            result[key] = (scores, "arithmetic_mean:MAE_only")
        else:
            raise ValueError("strategy must be adjudicated, single, majority, or mean")
    return result


def compare_run(store, run_id: str, strategy="adjudicated", within_distance=1,
                bootstrap_replicates=500, category_filter: list[str] | None = None,
                human_score_range: tuple[int, int] | None = None) -> dict:
    run = store.run(run_id)
    rubric = store.rubric(run["rubric_key"])
    humans = human_reference(store, strategy)
    items = {x.evaluation_id: x for x in store.run_items(run_id)}
    # For repeated runs, only first replicate enters cross-item human comparisons.
    results = {x["evaluation_id"]: x for x in store.results(run_id) if x["repeat_index"] == 0 and x["state"] == "completed"}
    by_criterion = {}
    for c in rubric.criteria:
        if not c.enabled:
            continue
        pairs = []
        by_category = defaultdict(list)
        for id_, item in items.items():
            if category_filter is not None and item.category not in category_filter:
                continue
            human_scores = humans.get((id_, rubric.key), ({}, "none"))[0]
            result = results.get(id_)
            judge_scores = ({j["criterion"]: j["score"]
                             for j in json.loads(result["payload"])["criteria"]}
                            if result else {})
            triplet = (id_, human_scores.get(c.key), judge_scores.get(c.key))
            if human_score_range is not None and triplet[1] is not None and not human_score_range[0] <= triplet[1] <= human_score_range[1]:
                continue
            pairs.append(triplet)
            by_category[item.category].append(triplet)
        if strategy == "mean":
            eligible = [(h, j) for _, h, j in pairs if h is not None and j is not None]
            by_criterion[c.key] = {"eligible": len(eligible), "mae": float(np.mean([abs(j-h) for h,j in eligible])) if eligible else None,
                                   "bias": float(np.mean([j-h for h,j in eligible])) if eligible else None,
                                   "reference_strategy": "human_mean:MAE_only"}
        else:
            metric = agreement(pairs, sorted(c.anchors), c.scale, within_distance, bootstrap_replicates)
            metric["by_category"] = {cat: agreement(p, sorted(c.anchors), c.scale, within_distance, 0)
                                     for cat, p in by_category.items()}
            metric["reference_strategy"] = strategy
            by_criterion[c.key] = metric
    return {"run_id": run_id, "rubric_key": rubric.key, "dataset_version": run["dataset_version"],
            "reference_strategy": strategy, "criteria": by_criterion,
            "note": "Synthetic fixtures must not be interpreted as empirical human agreement; repeated judge samples are excluded."}


def human_human(store, rubric_key: str, bootstrap_replicates=500) -> dict:
    groups = defaultdict(list)
    for a in store.latest_annotations(submitted_only=True):
        if a["rubric_key"] == rubric_key:
            groups[a["evaluation_id"]].append(a)
    rubric = store.rubric(rubric_key)
    by_criterion = {}
    for c in rubric.criteria:
        if not c.enabled:
            continue
        pairs = []
        for item_id, rows in groups.items():
            rows = sorted(rows, key=lambda x: x["annotator_id"])
            if len(rows) < 2:
                continue
            for i in range(len(rows)):
                for j in range(i+1, len(rows)):
                    pairs.append((item_id, _human_scores(rows[i]).get(c.key),
                                  _human_scores(rows[j]).get(c.key)))
        by_criterion[c.key] = agreement(pairs, sorted(c.anchors), c.scale,
                                        bootstrap_replicates=bootstrap_replicates)
        by_criterion[c.key]["unit"] = "human-pair within item; bootstrap clusters by item"
    return by_criterion


def compare_runs_holdout(store, run_a: str, run_b: str, strategy="adjudicated",
                         bootstrap_replicates=500, seed=42) -> dict:
    """Paired change in exact agreement on untouched holdout examples only.

    Reject incomparable criterion definitions, and don't silently tune on an in-sample
    evaluation set. A positive delta does not alone demonstrate significant improvement.
    """
    a, b = store.run(run_a), store.run(run_b)
    rub_a, rub_b = store.rubric(a["rubric_key"]), store.rubric(b["rubric_key"])
    if a["dataset_version"] != b["dataset_version"]:
        raise ValueError("run comparison requires the same versioned dataset")
    ids_a = {x.evaluation_id for x in store.run_items(run_a) if x.split == "holdout"}
    ids_b = {x.evaluation_id for x in store.run_items(run_b) if x.split == "holdout"}
    ids = sorted(ids_a & ids_b)
    if not ids:
        raise ValueError("no shared items explicitly marked split='holdout'")
    results_a = {x["evaluation_id"]: x for x in store.results(run_a)
                 if x["repeat_index"] == 0 and x["state"] == "completed"}
    results_b = {x["evaluation_id"]: x for x in store.results(run_b)
                 if x["repeat_index"] == 0 and x["state"] == "completed"}
    humans = human_reference(store, strategy)
    differences = {}
    for ca in rub_a.criteria:
        if not ca.enabled:
            continue
        cb = next((x for x in rub_b.criteria if x.key == ca.key and x.enabled), None)
        if cb is None or ca.model_dump(exclude={"weight", "critical_below"}) != cb.model_dump(exclude={"weight", "critical_below"}):
            continue
        records = []
        for item_id in ids:
            ha = humans.get((item_id, rub_a.key), ({}, "none"))[0].get(ca.key)
            hb = humans.get((item_id, rub_b.key), ({}, "none"))[0].get(cb.key)
            # Identical criterion definitions can use the same reference labels across rubrics,
            # but only if both human references are recorded and agree.
            if ha is None or hb is None or ha != hb or item_id not in results_a or item_id not in results_b:
                continue
            xa = next(x["score"] for x in json.loads(results_a[item_id]["payload"])["criteria"] if x["criterion"] == ca.key)
            xb = next(x["score"] for x in json.loads(results_b[item_id]["payload"])["criteria"] if x["criterion"] == ca.key)
            if xa is None or xb is None:
                continue
            records.append((item_id, float(xa == ha), float(xb == ha)))
        delta = float(np.mean([new-old for _,old,new in records])) if records else None
        ci = bootstrap_ci(records, lambda old,new: np.mean(new-old), bootstrap_replicates, seed)
        differences[ca.key] = {"n_shared_holdout": len(records), "delta_exact_b_minus_a": delta,
                               "delta_ci": ci, "identical_criterion_definitions": True}
    return {"run_a": run_a, "run_b": run_b, "holdout_item_ids": len(ids),
            "per_criterion": differences,
            "caution": "Compared only shared holdout items, identical criterion definitions and matching independently collected human references; zero overlapping valid scores produce N/A."}
