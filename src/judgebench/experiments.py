"""Paid controlled comparisons, with explicit provenance and no synthetic findings."""
import json
from collections import defaultdict, Counter

VALID_EXPERIMENTS = {"verbosity", "presentation", "injection", "attribution", "reference", "repeat"}


def paired_experiment(store, judge, item_id: str, kind: str,
                      variant_answer: str | None = None, runs: int = 2):
    if kind not in VALID_EXPERIMENTS:
        raise ValueError("unknown robustness experiment")
    item = store.item(item_id)
    rubric = store.rubric(item.rubric_key)
    if kind not in {"repeat", "reference"} and not variant_answer:
        raise ValueError("controlled variant answer is required")
    if runs < 1 or runs > 20:
        raise ValueError("runs must be between 1 and 20")
    config = judge.cfg
    baseline, altered = [], []
    for _ in range(runs):
        baseline.append(judge.evaluate(item, rubric))
        if kind == "reference":
            # Both reference answer and evidence removed in paired test.
            updated = item.model_copy(update={"reference_answer": None, "evidence": None})
            altered.append(judge.evaluate(updated, rubric))
        elif kind != "repeat":
            updated = item.model_copy(update={"answer": variant_answer})
            altered.append(judge.evaluate(updated, rubric))
    def compact(values):
        return [{"scores": {c.criterion: c.score for c in v["judgment"].criteria},
                 "rationales": {c.criterion: c.rationale for c in v["judgment"].criteria},
                 "tokens_in": v["tokens_in"], "tokens_out": v["tokens_out"]}
                for v in values]
    a, b = compact(baseline), compact(altered)
    deltas = defaultdict(list)
    for x, y in zip(a, b):
        for criterion in x["scores"]:
            sx, sy = x["scores"][criterion], y["scores"][criterion]
            if sx is not None and sy is not None:
                deltas[criterion].append(sy-sx)
    summary = {key: {"mean_delta": sum(vals)/len(vals), "n_pairs": len(vals),
                     "changed_count": sum(v != 0 for v in vals)}
               for key, vals in deltas.items()}
    if kind == "repeat":
        summary["repeat_rationale_variation"] = {
            key: len({v["rationales"][key] for v in a}) for key in a[0]["rationales"]}
        summary["repeat_score_variation"] = {
            key: len({v["scores"][key] for v in a}) for key in a[0]["scores"]}
    payload = {"kind": kind, "model": config.model, "rubric_key": rubric.key,
               "item_id": item_id, "runs": runs,
               "variant_answer": variant_answer if kind != "repeat" else None,
               "baseline": a, "variant": b, "summary": summary,
               "caution": "No causality without controlled equivalence and sufficient independently sampled items."}
    store.add_experiment(kind, payload, evaluation_id=item_id)
    return payload


def stability(store, run_id: str) -> dict:
    groups = defaultdict(list)
    for row in store.results(run_id):
        if row["state"] == "completed":
            groups[row["evaluation_id"]].append(json.loads(row["payload"]))
    results = {}
    for item, ratings in groups.items():
        if len(ratings) < 2:
            continue
        keys = {c["criterion"] for c in ratings[0]["criteria"]}
        results[item] = {}
        for criterion in keys:
            cells = [next(c for c in r["criteria"] if c["criterion"] == criterion) for r in ratings]
            results[item][criterion] = {"n": len(cells), "unique_scores": len({c["score"] for c in cells}),
                                        "unique_rationales": len({c["rationale"] for c in cells}),
                                        "score_counts": dict(Counter(str(c["score"]) for c in cells))}
    return results
