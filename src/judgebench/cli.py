"""Command-line interface for every essential experiment workflow."""
import argparse
import json
import os
import sys
from pathlib import Path
import yaml
from .datasets import load_path, export_items
from .schemas import JudgeConfig, HumanScores
from .rubrics import load_rubric
from .storage import Store
from .metrics import compare_run, human_human, compare_runs_holdout
from .judge import ClaudeJudge, prompt_snapshot, run_evaluations, JudgeError
from .experiments import paired_experiment, stability
from .reporting import report_dict, render_report


def parser():
    p = argparse.ArgumentParser(prog="judgebench", description="Rubric-based Claude evaluation platform")
    p.add_argument("--db", default=os.environ.get("JUDGEBENCH_DB", "judgebench.sqlite3"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    r = sub.add_parser("rubric-check"); r.add_argument("--rubric", default="configs/rubric.yaml")
    r = sub.add_parser("rubric-add"); r.add_argument("path")
    r = sub.add_parser("dataset-validate"); r.add_argument("path")
    r = sub.add_parser("dataset-import"); r.add_argument("path")
    r = sub.add_parser("dataset-export"); r.add_argument("--version"); r.add_argument("--format", choices=["csv", "jsonl"], default="jsonl"); r.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--dataset", required=True); r.add_argument("--rubric-key", default="default@1.0.0")
    r.add_argument("--config", default="configs/judge.yaml"); r.add_argument("--model"); r.add_argument("--limit", type=int)
    r.add_argument("--repeats", type=int); r.add_argument("--batch-size", type=int)
    r.add_argument("--unstructured-fallback", action="store_true")
    r = sub.add_parser("resume"); r.add_argument("run_id")
    r = sub.add_parser("cancel"); r.add_argument("run_id")
    r = sub.add_parser("list-runs")
    r = sub.add_parser("annotate"); r.add_argument("evaluation_id"); r.add_argument("--annotator", required=True)
    r.add_argument("--scores-json", required=True); r.add_argument("--comments", default=""); r.add_argument("--draft", action="store_true"); r.add_argument("--rubric-key")
    r = sub.add_parser("adjudicate"); r.add_argument("evaluation_id"); r.add_argument("--adjudicator", required=True)
    r.add_argument("--scores-json", required=True); r.add_argument("--reason", required=True); r.add_argument("--rubric-key")
    r = sub.add_parser("agreement"); r.add_argument("--run", required=True); r.add_argument("--strategy", choices=["adjudicated", "single", "majority", "mean"], default="adjudicated")
    r.add_argument("--bootstrap", type=int, default=500)
    r = sub.add_parser("compare-runs"); r.add_argument("--a", required=True); r.add_argument("--b", required=True); r.add_argument("--strategy", default="adjudicated", choices=["adjudicated", "single", "majority"])
    r = sub.add_parser("human-agreement"); r.add_argument("--rubric-key", default="default@1.0.0")
    r = sub.add_parser("stability"); r.add_argument("--run", required=True)
    r = sub.add_parser("experiment"); r.add_argument("--item", required=True)
    r.add_argument("--kind", required=True, choices=["verbosity", "presentation", "injection", "attribution", "reference", "repeat"])
    r.add_argument("--variant-text"); r.add_argument("--runs", type=int, default=2); r.add_argument("--config", default="configs/judge.yaml")
    r = sub.add_parser("report"); r.add_argument("--run", required=True); r.add_argument("--strategy", choices=["adjudicated", "single", "majority", "mean"], default="adjudicated")
    r.add_argument("--format", choices=["json", "csv", "html", "markdown"], default="markdown"); r.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        store = Store(args.db)
        if args.cmd == "init":
            rubric = load_rubric("configs/rubric.yaml")
            store.add_rubric(rubric)
            print("initialized", store.path, rubric.key)
        elif args.cmd == "rubric-check":
            rubric = load_rubric(args.rubric)
            print("validated", rubric.key, "criteria", len(rubric.criteria))
        elif args.cmd == "rubric-add":
            rubric = load_rubric(args.path)
            store.add_rubric(rubric)
            print("stored immutable rubric", rubric.key)
        elif args.cmd == "dataset-validate":
            rows = load_path(args.path)
            print("valid", len(rows), "records; versions", sorted({r.dataset_version for r in rows}))
        elif args.cmd == "dataset-import":
            rows = load_path(args.path)
            print("imported", store.add_items(rows), "records")
        elif args.cmd == "dataset-export":
            Path(args.out).write_bytes(export_items(store.items(args.version), args.format))
            print("exported", args.out)
        elif args.cmd == "run":
            cfg = JudgeConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
            changes = {"model": args.model, "limit": args.limit, "repeat_count": args.repeats, "batch_size": args.batch_size, "structured_output": False if args.unstructured_fallback else None}
            cfg = cfg.model_copy(update={k: v for k, v in changes.items() if v is not None})
            cfg = JudgeConfig.model_validate(cfg.model_dump())
            rubric = store.rubric(args.rubric_key)
            if not store.items(args.dataset):
                raise ValueError("no matching items to evaluate")
            judge = ClaudeJudge(cfg)  # availability check before creating a run
            run_id = store.create_run(cfg, rubric, args.dataset, prompt_snapshot(cfg.prompt_version))
            print("run_id", run_id, flush=True)
            print(run_evaluations(store, run_id, judge, on_progress=lambda a,b: print(f"saved {a}/{b}", flush=True)))
        elif args.cmd == "resume":
            run = store.run(args.run_id)
            cfg = JudgeConfig.model_validate_json(run["config"])
            judge = ClaudeJudge(cfg)
            print(run_evaluations(store, args.run_id, judge))
        elif args.cmd == "cancel":
            store.set_run_state(args.run_id, "cancelled")
            print("cancel requested; stops before next API call")
        elif args.cmd == "list-runs":
            print(json.dumps(store.runs(), indent=2))
        elif args.cmd == "annotate":
            scores = json.loads(args.scores_json)
            result = store.annotate(args.evaluation_id, args.annotator, HumanScores(scores=scores, comments=args.comments), "draft" if args.draft else "submitted", rubric_key=args.rubric_key)
            print("saved human annotation", result)
        elif args.cmd == "adjudicate":
            store.adjudicate(args.evaluation_id, args.adjudicator, HumanScores(scores=json.loads(args.scores_json)), args.reason, rubric_key=args.rubric_key)
            print("adjudication saved")
        elif args.cmd == "agreement":
            print(json.dumps(compare_run(store, args.run, args.strategy, bootstrap_replicates=args.bootstrap), indent=2))
        elif args.cmd == "compare-runs":
            print(json.dumps(compare_runs_holdout(store, args.a, args.b, args.strategy), indent=2))
        elif args.cmd == "human-agreement":
            print(json.dumps(human_human(store, args.rubric_key), indent=2))
        elif args.cmd == "stability":
            print(json.dumps(stability(store, args.run), indent=2))
        elif args.cmd == "experiment":
            cfg = JudgeConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
            judge = ClaudeJudge(cfg)
            print(json.dumps(paired_experiment(store, judge, args.item, args.kind, args.variant_text, args.runs)["summary"], indent=2))
        elif args.cmd == "report":
            payload = report_dict(store, args.run, args.strategy)
            Path(args.out).write_bytes(render_report(payload, args.format))
            print("report written", args.out)
        return 0
    except (ValueError, OSError, JudgeError, ImportError) as exc:
        # No raw exception body from Anthropic is leaked to the terminal.
        print("Error:", str(exc) if not isinstance(exc, JudgeError) else f"{exc.category}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
