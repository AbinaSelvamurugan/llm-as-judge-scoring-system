"""Reproducible summaries based solely on persisted results, not claimed performance."""
from __future__ import annotations
import csv
import html
import io
import json
from .metrics import compare_run, human_human
from .experiments import stability


def report_dict(store, run_id: str, strategy="adjudicated", n_boot=500):
    run = store.run(run_id)
    rubric = store.rubric(run["rubric_key"])
    run_results = store.results(run_id)
    items = store.run_items(run_id)
    annotations = store.latest_annotations(submitted_only=True, rubric_key=rubric.key)
    ids = {i.evaluation_id for i in items}
    cfg = json.loads(run["config"])
    comparisons = compare_run(store, run_id, strategy, bootstrap_replicates=n_boot)
    return {"title": "JudgeBench research report", "research_status": "OBSERVATIONAL; verify human provenance",
            "run": run, "rubric": rubric.model_dump(mode="json"),
            "sample": {"dataset_items": len(items), "judge_successes": sum(r["state"] == "completed" for r in run_results),
                       "judge_failures": sum(r["state"] != "completed" for r in run_results),
                       "submitted_annotations": sum(a["evaluation_id"] in ids for a in annotations),
                       "dataset_categories": {cat: sum(i.category == cat for i in items) for cat in sorted({i.category for i in items})}},
            "agreement": comparisons, "human_human": human_human(store, rubric.key, n_boot),
            "stability": stability(store, run_id),
            "experiments": [x for x in store.experiments()
                            if x["evaluation_id"] in ids and
                            json.loads(x["payload"]).get("rubric_key") == rubric.key and
                            json.loads(x["payload"]).get("model") == cfg["model"]],
            "errors": [{"evaluation_id": r["evaluation_id"], "error_type": r["error_type"]} for r in run_results if r["state"] != "completed"],
            "usage": {"input_tokens": sum(r["tokens_in"] or 0 for r in run_results),
                      "output_tokens": sum(r["tokens_out"] or 0 for r in run_results),
                      "cost": None},
            "method": {"human_reference": strategy,
                       "missing": "pairwise exclusion of null scores, omitted ungraded cases reported separately",
                       "uncertainty": f"cluster bootstrap over evaluation IDs; {n_boot} replicates",
                       "repetitions": "repeat_index 0 only for human–judge metrics"},
            "limitations": ["Human labels are judgments, not infallible ground truth.",
                            "Correlation measures ranking, not agreement.",
                            "No reliability inference without independent human annotations.",
                            "Small samples, category imbalance and source evidence gaps limit conclusions.",
                            "API model aliases may change behavior even for identical configuration."]}


def markdown_report(payload: dict) -> str:
    run = payload["run"]
    lines = ["# JudgeBench research report", "", "## Abstract", "",
             "Rubric-based automated judgments are compared with available blinded human annotations. "
             "Absent human references produce unavailable statistics; no reliability level is assumed.", "",
             "## Methodology and configuration", "",
             f"- Run ID: `{run['run_id']}`", f"- Dataset: `{run['dataset_version']}`",
             f"- Rubric: `{run['rubric_key']}`", f"- Configuration: `{run['config']}`",
             f"- Human reference: `{payload['method']['human_reference']}`", "",
             "## Dataset and annotation", "",
             f"- Items: {payload['sample']['dataset_items']}",
             f"- Successful provider judgments: {payload['sample']['judge_successes']}",
             f"- Provider failures: {payload['sample']['judge_failures']}",
             f"- Submitted annotations: {payload['sample']['submitted_annotations']}",
             f"- Category distribution: {payload['sample']['dataset_categories']}", "",
             "## Criterion-level agreement", "",
             "| Criterion | Eligible | Exact | Within-one | Kappa | Quadratic kappa | MAE | Signed bias | Exact 95% CI |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|" ]
    def disp(x):
        return "N/A" if x is None else (f"{x:.3f}" if isinstance(x, (float, int)) else str(x))
    for key, stats in payload["agreement"]["criteria"].items():
        ci = stats.get("exact_ci")
        cells = [key, stats.get("eligible"), stats.get("exact"), stats.get("within_one"),
                 stats.get("kappa"), stats.get("weighted_kappa"), stats.get("mae"), stats.get("bias")]
        lines.append("| " + " | ".join(disp(c) for c in cells) + " | " +
                     ("–" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]") + " |")
    lines += ["", "## Confusion matrices", "", "Rows = human, columns = Claude; count labels in ascending order.", ""]
    for key, stats in payload["agreement"]["criteria"].items():
        lines.extend([f"### {key}", "", f"Labels: {stats.get('labels', 'N/A')}",
                      "", f"Counts: `{json.dumps(stats.get('confusion_counts'))}`", ""])
    lines += ["## Human–human reliability", "",
              json.dumps({k: {"eligible": v["eligible"], "exact": v["exact"],
                              "weighted_kappa": v["weighted_kappa"]} for k, v in payload["human_human"].items()}, indent=2), "",
              "## Robustness and stability", "", f"Stability: `{json.dumps(payload['stability'])}`", "",
              "## API usage", "", f"Known token totals: {payload['usage']}; monetary cost unavailable without verified pricing.", "",
              "## Limitations", ""]
    lines += [f"- {x}" for x in payload["limitations"]]
    lines += ["", "## Recommendations", "", "Human-review low factual/grounding scores, abstentions, and large individual disagreements. Calibrate thresholds only on development data, then evaluate untouched holdouts.", "",
              "## Research manuscript outline", "", "Abstract; Introduction; Methods; Datasets; Annotation protocol; Rubric and anchors; Results; Error analysis; Discussion; Limitations; References (verified sources only).", ""]
    return "\n".join(lines)


def render_report(payload: dict, fmt="markdown") -> bytes:
    if fmt == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False).encode()
    if fmt == "markdown":
        return markdown_report(payload).encode()
    if fmt == "html":
        return ("<!doctype html><html><head><meta charset='utf-8'><title>JudgeBench Report</title></head>"
                "<body><pre>" + html.escape(markdown_report(payload)) + "</pre></body></html>").encode()
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["criterion", "eligible", "exact", "within_one", "kappa", "weighted_kappa", "mae", "bias"])
        for key, s in payload["agreement"]["criteria"].items():
            writer.writerow([key] + [s.get(k) for k in ["eligible", "exact", "within_one", "kappa", "weighted_kappa", "mae", "bias"]])
        return buf.getvalue().encode()
    raise ValueError("format must be json, markdown, html or csv")
