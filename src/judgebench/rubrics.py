"""Rubric integrity, aggregation and YAML loading."""
import json
from pathlib import Path
import yaml
from .schemas import Rubric, ProviderJudgment, Aggregate


def load_rubric(path: str | Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def validate_scores(rubric: Rubric, scores: dict[str, int | None]) -> None:
    active = {c.key: c for c in rubric.criteria if c.enabled}
    if set(scores) != set(active):
        raise ValueError(f"criteria mismatch: missing={set(active)-set(scores)}, extra={set(scores)-set(active)}")
    for key, value in scores.items():
        criterion = active[key]
        if value is None and not criterion.allow_abstain:
            raise ValueError(f"{key}: cannot abstain")
        if value is not None and (type(value) is not int or value not in criterion.anchors):
            raise ValueError(f"{key}: invalid score {value}")


def validate_judgment(rubric: Rubric, item, judgment: ProviderJudgment) -> None:
    if judgment.evaluation_id != item.evaluation_id or judgment.rubric_version != rubric.key:
        raise ValueError("judge returned wrong evaluation_id or rubric version")
    keys = [x.criterion for x in judgment.criteria]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate criterion result")
    validate_scores(rubric, {c.criterion: c.score for c in judgment.criteria})
    for field in judgment.criteria:
        if field.answer_excerpt and " ".join(field.answer_excerpt.casefold().split()) not in " ".join(item.answer.casefold().split()):
            raise ValueError("answer excerpt is not present in evaluated answer")
        if field.evidence_locator and item.evidence and " ".join(field.evidence_locator.casefold().split()) not in " ".join(item.evidence.casefold().split()):
            raise ValueError("evidence locator is not present in supplied source")
    for c in rubric.criteria:
        if c.enabled and c.evidence_requirement == "required" and not item.evidence:
            scored = next(x for x in judgment.criteria if x.criterion == c.key)
            if scored.score is not None:
                raise ValueError(f"{c.key}: must abstain without required evidence")
    if not item.evidence and any(c.evidence_locator for c in judgment.criteria):
        raise ValueError("invented evidence locator without evidence source")


def aggregate(rubric: Rubric, scores: dict[str, int | None]) -> Aggregate:
    validate_scores(rubric, scores)
    active = [c for c in rubric.criteria if c.enabled]
    normalized = []
    critical = []
    abstained = []
    for c in active:
        score = scores[c.key]
        if score is None:
            abstained.append(c.key)
            continue
        minimum, maximum = min(c.anchors), max(c.anchors)
        normalized.append(((score - minimum) / (maximum - minimum), c.weight))
        if c.critical_below is not None and score <= c.critical_below:
            critical.append(c.key)
    mean = sum(v for v, _ in normalized) / len(normalized) if normalized else None
    weight_total = sum(w for _, w in normalized)
    weighted = sum(v * w for v, w in normalized) / weight_total if weight_total else None
    if rubric.fail_on_critical and critical:
        decision = "fail"
    elif not normalized or (rubric.pass_threshold is not None and abstained):
        decision = "not_assessable"
    elif rubric.pass_threshold is None:
        decision = "not_defined"
    else:
        decision = "pass" if weighted is not None and weighted >= rubric.pass_threshold else "fail"
    return Aggregate(unweighted_normalized=mean, weighted_normalized=weighted,
                     critical_failure=bool(critical), critical_criteria=critical,
                     missing_criteria=[], abstained_criteria=abstained, decision=decision)


def judgment_scores(judgment: ProviderJudgment) -> dict[str, int | None]:
    return {x.criterion: x.score for x in judgment.criteria}


def rubric_json(rubric: Rubric) -> str:
    return json.dumps(rubric.model_dump(mode="json"), sort_keys=True)
