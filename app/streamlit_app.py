"""JudgeBench Streamlit research workbench; all panels use persisted records."""
from __future__ import annotations
import json
import os
import secrets
import sys
from pathlib import Path

# Launch from repository root: streamlit run app/streamlit_app.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from judgebench.datasets import load_bytes, export_items
from judgebench.judge import ClaudeJudge, JudgeError, prompt_snapshot, run_evaluations
from judgebench.metrics import compare_run, human_human, compare_runs_holdout
from judgebench.experiments import paired_experiment, stability
from judgebench.reporting import report_dict, render_report
from judgebench.rubrics import load_rubric
from judgebench.schemas import Rubric, HumanScores, JudgeConfig
from judgebench.storage import Store

st.set_page_config(page_title="JudgeBench", page_icon="⚖️", layout="wide")


def message_error(exc: Exception):
    # Avoid leaking provider request, path or private environment values.
    if isinstance(exc, JudgeError):
        st.error(f"Judge service failed: {exc.category}")
    elif isinstance(exc, (ValueError, KeyError)):
        st.error(str(exc)[:400])
    else:
        st.error("Operation failed; check server logs and configuration.")


def auth():
    annotator_token, reviewer_token = os.getenv("JUDGEBENCH_ANNOTATOR_TOKEN"), os.getenv("JUDGEBENCH_REVIEW_TOKEN")
    if not (annotator_token and reviewer_token):
        st.sidebar.warning("Local mode: access controls are not enabled. Do not host for blinded studies.")
        return "reviewer"
    if "role" not in st.session_state:
        with st.form("login"):
            password = st.text_input("Study access token", type="password")
            entered = st.form_submit_button("Sign in")
            if entered:
                if secrets.compare_digest(password, reviewer_token):
                    st.session_state.role = "reviewer"
                    st.rerun()
                elif secrets.compare_digest(password, annotator_token):
                    st.session_state.role = "annotator"
                    st.rerun()
                else:
                    st.error("Invalid token")
        st.stop()
    return st.session_state.role


role = auth()
store = Store()
st.title("JudgeBench")
st.caption("Rubric-based evaluation of LLM answers against independently collected human judgments")
ALL_PAGES = ["Overview", "Dataset Management", "Rubric Builder", "Judge Configuration",
             "Run Evaluations", "Human Annotation", "Agreement Analysis",
             "Disagreement Explorer", "Bias & Robustness", "Reports & Export"]
pages = ALL_PAGES if role == "reviewer" else ["Human Annotation"]
page = st.sidebar.radio("Navigation", pages)
st.sidebar.caption(f"Role: {role}. SQLite: configured local store")


def rubric_options():
    return [r.key for r in store.rubrics()]


def run_options():
    return [r["run_id"] for r in store.runs()]


def must_have(items, label):
    if not items:
        st.info(f"No {label} yet. Import a dataset or create a run as appropriate.")
        st.stop()


def render_item(item):
    st.markdown("**Question**")
    st.write(item.question)
    st.markdown("**Answer to evaluate**")
    st.write(item.answer)
    st.markdown("**Reference answer**")
    st.write(item.reference_answer or "Not provided")
    st.markdown("**Evidence/context**")
    st.write(item.evidence or "Not provided")
    st.caption(f"Category: {item.category} | Dataset: {item.dataset_version} | Rubric: {item.rubric_key}")


def get_config():
    if "judge_config" not in st.session_state:
        path = Path(__file__).resolve().parents[1] / "configs" / "judge.yaml"
        st.session_state.judge_config = yaml.safe_load(path.read_text())
    return st.session_state.judge_config


def preview_usage(cfg, n_items):
    count = min(n_items, cfg.limit) * cfg.repeat_count
    past = [r for r in store.results() if r["tokens_in"] is not None and r["tokens_out"] is not None]
    st.write(f"Estimated requests: **{count}** (maximum, before resuming).")
    if past:
        st.write("Historical mean tokens per request, as measured (not a guarantee):",
                 round(sum(r["tokens_in"] for r in past)/len(past)), "input and",
                 round(sum(r["tokens_out"] for r in past)/len(past)), "output")
    else:
        st.info("No provider token history: token forecast and monetary cost unavailable.")


if page == "Overview":
    overview = store.summary()
    cols = st.columns(4)
    for col, (name, key) in zip(cols, [("Dataset records", "items"), ("Completed Claude judgments", "judgments"),
                                       ("Human-labelled items", "annotated_items"), ("Provider failures", "api_failures")]):
        col.metric(name, overview[key])
    st.metric("Human annotation coverage", f"{100 * overview['annotated_items']/overview['items']:.1f}%" if overview["items"] else "N/A")
    runs = store.runs()
    if not runs:
        st.info("Empty workspace. Add the default rubric, import task-answer records, then run Claude and collect blind human ratings.")
    else:
        selected = st.selectbox("Analysis run", [r["run_id"] for r in runs])
        stats = compare_run(store, selected, strategy="single", bootstrap_replicates=100)
        eligible = sum(s.get("eligible", 0) for s in stats["criteria"].values())
        st.metric("Eligible criterion-level comparisons (single-human reference)", eligible)
        if eligible:
            st.dataframe(pd.DataFrame([{"criterion": k, "n": v["eligible"],
                                        "exact": v["exact"], "within_one": v["within_one"],
                                        "kappa": v["kappa"], "mae": v["mae"]}
                                       for k, v in stats["criteria"].items()]), hide_index=True)
        else:
            st.info("Agreement unavailable until matching independent submitted human annotations exist.")
    st.caption("Criterion-level counts must not be represented as independent item counts.")

elif page == "Dataset Management":
    st.subheader("Validated task-answer dataset")
    uploaded = st.file_uploader("Import CSV or JSONL (max 8 MB)", type=["csv", "jsonl"])
    if uploaded:
        try:
            new_items = load_bytes(uploaded.getvalue(), Path(uploaded.name).suffix)
            st.success(f"Validated {len(new_items)} records")
            st.dataframe(pd.DataFrame([x.model_dump() for x in new_items]).head(25))
            if st.button("Import validated records", type="primary"):
                st.success(f"Imported {store.add_items(new_items)} records")
                st.rerun()
        except Exception as exc:
            message_error(exc)
    items = store.items()
    if items:
        versions = sorted({x.dataset_version for x in items})
        version = st.selectbox("Dataset version filter", ["All"] + versions)
        cats = sorted({x.category for x in items})
        selected_cats = st.multiselect("Categories", cats, default=cats)
        filtered = [i for i in items if (version == "All" or i.dataset_version == version) and i.category in selected_cats]
        st.dataframe(pd.DataFrame([{**i.model_dump(), "human_submitted": any(a["evaluation_id"] == i.evaluation_id for a in store.latest_annotations(submitted_only=True))} for i in filtered]), hide_index=True)
        st.write("Unique records", len(filtered), "Categories", len({i.category for i in filtered}))
        fmt = st.radio("Export format", ["jsonl", "csv"], horizontal=True)
        st.download_button("Download filtered dataset", export_items(filtered, fmt),
                           file_name=f"judgebench_items.{fmt}", mime="text/plain")
    else:
        st.info("No records imported. Committed fixture is synthetic and unlabeled.")

elif page == "Rubric Builder":
    st.subheader("Rubric versions")
    options = rubric_options()
    if not options:
        if st.button("Install default rubric"):
            rubric = load_rubric(Path(__file__).resolve().parents[1] / "configs/rubric.yaml")
            store.add_rubric(rubric)
            st.rerun()
        st.stop()
    key = st.selectbox("Rubric", options)
    rubric = store.rubric(key)
    st.write(rubric.title, "—", rubric.key)
    st.dataframe(pd.DataFrame([{"criterion": c.key, "type": c.scale.value, "active": c.enabled,
                                "weight": c.weight, "critical at/below": c.critical_below,
                                "evidence": c.evidence_requirement} for c in rubric.criteria]), hide_index=True)
    for criterion in rubric.criteria:
        with st.expander(f"{criterion.name} — {criterion.scale.value}"):
            st.write(criterion.description)
            st.write(criterion.instructions)
            for score, definition in criterion.anchors.items():
                st.write(f"**{score}:** {definition}")
            st.caption(f"Abstain: {criterion.abstain_when}")
            st.write("Positive:", criterion.positive_example, "Negative:", criterion.negative_example)
    st.caption("Edit the complete rubric YAML: scores, anchors, weights, critical failures, examples, binary/ordinal scales, or optional criteria. A new version is required.")
    source = yaml.safe_dump(rubric.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    with st.form("rubric_edit"):
        edited = st.text_area("Rubric YAML", value=source, height=430)
        if st.form_submit_button("Validate and save new version"):
            try:
                changed = Rubric.model_validate(yaml.safe_load(edited))
                if changed.key == rubric.key:
                    raise ValueError("set a NEW rubric version before saving")
                store.add_rubric(changed)
                st.success(f"Stored {changed.key}. Existing judgments remain bound to their original version.")
            except Exception as exc:
                message_error(exc)
    with st.expander("Compare two rubric versions"):
        if len(options) > 1:
            other = st.selectbox("Compare against", [r for r in options if r != key])
            st.code(json.dumps({"selected": rubric.model_dump(mode="json"),
                                "other": store.rubric(other).model_dump(mode="json")}, indent=2))
        else:
            st.info("Create a second version to compare definitions.")

elif page == "Judge Configuration":
    st.subheader("Claude judge settings")
    options = rubric_options()
    items = store.items()
    datasets = sorted({i.dataset_version for i in items})
    must_have(options, "rubrics")
    must_have(datasets, "datasets")
    raw = get_config()
    with st.form("configuration"):
        model = st.text_input("Claude model ID (verified against provider on run)", value=raw["model"])
        prompt_version = st.selectbox("Prompt version", ["v1", "v2"])
        rubric_key = st.selectbox("Rubric version", options, index=options.index(st.session_state.get("selected_rubric", options[0])) if st.session_state.get("selected_rubric") in options else 0)
        dataset = st.selectbox("Dataset version", datasets)
        ref_policy = st.selectbox("Reference policy", ["as_available", "require_reference", "no_reference"], index=["as_available", "require_reference", "no_reference"].index(raw.get("reference_policy", "as_available")))
        evid_policy = st.selectbox("Evidence policy", ["as_available", "require_evidence", "no_evidence"], index=["as_available", "require_evidence", "no_evidence"].index(raw.get("evidence_policy", "as_available")))
        c1, c2, c3 = st.columns(3)
        limit = c1.number_input("Evaluation limit", min_value=1, value=int(raw.get("limit", 20)))
        repeats = c2.number_input("Repeats", min_value=1, max_value=20, value=int(raw.get("repeat_count", 1)))
        batch_size = c3.number_input("Batch size (sequential)", min_value=1, max_value=100, value=int(raw.get("batch_size", 5)))
        structured = st.checkbox("Use current structured JSON output (recommended)", value=raw.get("structured_output", True))
        timeout = st.number_input("API timeout seconds", min_value=1.0, max_value=600.0, value=float(raw.get("timeout_seconds", 60)))
        retries = st.number_input("Retries", min_value=0, max_value=8, value=int(raw.get("max_retries", 2)))
        max_tokens = st.number_input("Maximum output tokens", min_value=256, max_value=8192, value=int(raw.get("max_tokens", 2048)))
        submit = st.form_submit_button("Validate and save configuration")
        if submit:
            try:
                cfg = JudgeConfig(model=model, prompt_version=prompt_version, limit=limit,
                                  repeat_count=repeats, batch_size=batch_size, timeout_seconds=timeout,
                                  max_retries=retries, max_tokens=max_tokens, structured_output=structured,
                                  reference_policy=ref_policy, evidence_policy=evid_policy)
                subset = store.items(dataset)[:cfg.limit]
                if not subset:
                    raise ValueError("dataset version contains no records")
                if ref_policy == "require_reference" and any(not i.reference_answer for i in subset):
                    raise ValueError("some selected items lack reference answers")
                if evid_policy == "require_evidence" and any(not i.evidence for i in subset):
                    raise ValueError("some selected items lack evidence")
                st.session_state.judge_config = cfg.model_dump()
                st.session_state.selected_rubric = rubric_key
                st.session_state.selected_dataset = dataset
                st.session_state.config_saved = True
                st.success("Local configuration validated. Model access is checked at run start.")
            except Exception as exc:
                message_error(exc)
    cfg = JudgeConfig.model_validate(get_config())
    preview_usage(cfg, len(store.items(st.session_state.get("selected_dataset", datasets[0]))))

elif page == "Run Evaluations":
    st.subheader("Execute Claude judgments")
    must_have(store.items(), "dataset items")
    options = rubric_options()
    must_have(options, "rubrics")
    cfg = JudgeConfig.model_validate(get_config())
    rk = st.selectbox("Rubric", options, index=options.index(st.session_state.get("selected_rubric", options[0])) if st.session_state.get("selected_rubric") in options else 0)
    datasets = sorted({i.dataset_version for i in store.items()})
    must_have(datasets, "matching dataset versions")
    dataset = st.selectbox("Dataset", datasets)
    preview_usage(cfg, len(store.items(dataset)))
    if st.button("Start API evaluation", type="primary"):
        try:
            judge = ClaudeJudge(cfg)  # SDK and provider availability validated before run exists
            rubric = store.rubric(rk)
            run_id = store.create_run(cfg, rubric, dataset, prompt_snapshot(cfg.prompt_version))
            st.info(f"Run ID: {run_id}")
            progress = st.progress(0)
            def show(a, b):
                progress.progress(a/b)
            summary = run_evaluations(store, run_id, judge, on_progress=show)
            st.success(f"Run saved: {summary}") if not summary["failed"] else st.warning(f"Run partially failed: {summary}")
        except Exception as exc:
            message_error(exc)
    runs = store.runs()
    if runs:
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("run_id", "dataset_version", "rubric_key", "status", "created_at")} for r in runs]), hide_index=True)
        selected = st.selectbox("Inspect / control run", [r["run_id"] for r in runs])
        c1, c2 = st.columns(2)
        if c1.button("Resume unfinished or failed results"):
            try:
                run = store.run(selected)
                judge = ClaudeJudge(JudgeConfig.model_validate_json(run["config"]))
                st.info(str(run_evaluations(store, selected, judge)))
            except Exception as exc:
                message_error(exc)
        if c2.button("Request cancellation"):
            store.set_run_state(selected, "cancelled")
            st.warning("Cancellation requested. In-flight provider requests are not interrupted.")
        st.dataframe(pd.DataFrame(store.results(selected)), hide_index=True)

elif page == "Human Annotation":
    st.subheader("Blind independent human annotation")
    st.info("Automated judge scores and rationales are not shown on this page.")
    annotator = st.text_input("Pseudonymous annotator ID", value=st.session_state.get("annotator_id", ""))
    if not annotator.strip():
        st.info("Enter your assigned pseudonymous annotator ID to continue.")
        st.stop()
    st.session_state.annotator_id = annotator.strip()
    items = store.items()
    must_have(items, "dataset items")
    category = st.selectbox("Task category", ["All"] + sorted({x.category for x in items}))
    subset = [x for x in items if category == "All" or x.category == category]
    assigned_rubric = st.selectbox("Rubric version for annotation", rubric_options())
    last = {a["evaluation_id"]: a for a in store.latest_annotations(annotator_id=annotator, rubric_key=assigned_rubric)}
    status = st.selectbox("Status", ["Unlabelled / draft", "Submitted", "All"])
    if status == "Unlabelled / draft":
        subset = [x for x in subset if x.evaluation_id not in last or last[x.evaluation_id]["state"] == "draft"]
    elif status == "Submitted":
        subset = [x for x in subset if x.evaluation_id in last and last[x.evaluation_id]["state"] == "submitted"]
    must_have(subset, "matching annotation tasks")
    selected = st.selectbox("Evaluation ID", [x.evaluation_id for x in subset])
    item = store.item(selected)
    rubric = store.rubric(assigned_rubric)
    render_item(item)
    existing = json.loads(last[selected]["scores"]) if selected in last else {"scores": {}, "comments": "", "insufficient_evidence": False}
    with st.form(f"annotation_{annotator}_{selected}"):
        responses = {}
        for c in rubric.criteria:
            if not c.enabled:
                continue
            st.markdown(f"**{c.name}** — {c.description}")
            st.caption(c.instructions + " Abstain when: " + c.abstain_when)
            labels = ["Unrated"] + [f"{v}: {a}" for v, a in sorted(c.anchors.items())]
            if c.allow_abstain:
                labels += ["Cannot assess"]
            old = existing["scores"].get(c.key, "Unrated")
            ix = labels.index(f"{old}: {c.anchors[old]}") if isinstance(old, int) and old in c.anchors else (labels.index("Cannot assess") if old is None and c.key in existing["scores"] and "Cannot assess" in labels else 0)
            choice = st.selectbox(c.name, labels, index=ix, key=f"score_{annotator}_{selected}_{c.key}")
            responses[c.key] = "unrated" if choice == "Unrated" else (None if choice == "Cannot assess" else int(choice.split(":", 1)[0]))
        comments = st.text_area("Optional evidence-based explanation / uncertainties", value=existing["comments"])
        insufficient = st.checkbox("Ambiguous task or insufficient reference evidence", value=existing["insufficient_evidence"])
        draft = st.form_submit_button("Save draft")
        submit = st.form_submit_button("Submit independent annotation")
        if draft or submit:
            try:
                if submit and any(v == "unrated" for v in responses.values()):
                    raise ValueError("Complete each criterion or select Cannot assess before submitting")
                scores = {k: v for k, v in responses.items() if v != "unrated"}
                store.annotate(selected, annotator.strip(), HumanScores(scores=scores, comments=comments,
                                insufficient_evidence=insufficient), state="submitted" if submit else "draft", rubric_key=rubric.key)
                st.success("Submitted; all revisions are retained." if submit else "Draft saved.")
                st.rerun()
            except Exception as exc:
                message_error(exc)
    if selected in last and last[selected]["state"] == "submitted":
        st.caption("Resubmitting creates an auditable new revision; original submitted rating remains preserved.")

elif page == "Agreement Analysis":
    st.subheader("Judge–human and human–human agreement")
    all_run_rows = store.runs()
    must_have(all_run_rows, "judge runs")
    rubrics = sorted({r["rubric_key"] for r in all_run_rows})
    models = sorted({JudgeConfig.model_validate_json(r["config"]).model for r in all_run_rows})
    selected_rubric = st.selectbox("Rubric version filter", rubrics)
    selected_model = st.selectbox("Judge model filter", models)
    runs = [r["run_id"] for r in all_run_rows if r["rubric_key"] == selected_rubric and JudgeConfig.model_validate_json(r["config"]).model == selected_model]
    must_have(runs, "runs for this rubric and model")
    run_id = st.selectbox("Run", runs)
    task_categories = sorted({i.category for i in store.run_items(run_id)})
    selected_categories = st.multiselect("Task category filter", task_categories, default=task_categories)
    human_range = st.slider("Human score range (inclusive)", min_value=0, max_value=5, value=(0,5))
    strategy = st.selectbox("Human reference (explicitly defined)", ["adjudicated", "single", "majority", "mean"])
    distance = st.slider("Ordinal tolerance (rating points)", 0, 3, 1)
    boot = st.number_input("Bootstrap replicates", min_value=0, max_value=5000, value=300, step=100)
    stats = compare_run(store, run_id, strategy, distance, boot, selected_categories, human_range)
    human_stats = human_human(store, stats["rubric_key"], boot)
    criterion = st.selectbox("Criterion", list(stats["criteria"]))
    res = stats["criteria"][criterion]
    if not res.get("eligible"):
        st.info("No eligible paired human and provider scores for the selected comparison.")
    else:
        cols = st.columns(4)
        for col, (name, field) in zip(cols, [("Eligible", "eligible"), ("Exact", "exact"), ("Within tolerance", "within_one"), ("Mean absolute error", "mae")]):
            v = res.get(field)
            col.metric(name, f"{v:.3f}" if isinstance(v, float) else (str(v) if v is not None else "N/A"))
        st.json({k: v for k,v in res.items() if k not in ("by_category", "score_differences")})
        if res.get("confusion_counts"):
            labels = [str(x) for x in res["labels"]]
            fig = px.imshow(res["confusion_counts"], x=labels, y=labels, text_auto=True,
                            labels={"x": "Claude predicted score", "y": "Human reference score", "color": "Cases"})
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(px.histogram(pd.DataFrame({"Claude minus human": res["score_differences"]}),
                                         x="Claude minus human"), use_container_width=True)
        st.write("Category breakdown (small samples require care):")
        st.dataframe(pd.DataFrame([{"category": k, "eligible": v["eligible"], "exact": v["exact"], "mae": v["mae"]}
                                  for k,v in res.get("by_category", {}).items()]), hide_index=True)
    with st.expander("Compare prompt/rubric runs on a held-out split"):
        candidates = [r for r in runs if r != run_id]
        if candidates:
            other = st.selectbox("Second run", candidates)
            if st.button("Calculate paired held-out comparison"):
                try:
                    st.json(compare_runs_holdout(store, run_id, other, strategy if strategy != "mean" else "single"))
                except Exception as exc:
                    message_error(exc)
        else:
            st.info("At least two runs are necessary. Records must explicitly mark split='holdout'.")
    with st.expander("Human–human comparison on co-rated tasks"):
        st.json(human_stats)
    st.caption("Abstentions/missingness excluded pairwise and counted separately. Multiple ratings of one item are correlated; CIs cluster by item. Mean reference supports MAE/bias only.")

elif page == "Disagreement Explorer":
    st.subheader("Review disagreements and adjudicate")
    runs = run_options()
    must_have(runs, "judge runs")
    run_id = st.selectbox("Run", runs)
    rows = [r for r in store.results(run_id) if r["state"] == "completed" and r["repeat_index"] == 0]
    must_have(rows, "completed judgments")
    current_rubric = store.run(run_id)["rubric_key"]
    anno = store.latest_annotations(submitted_only=True, rubric_key=current_rubric)
    scored = []
    for row in rows:
        judge = json.loads(row["payload"])
        r = [a for a in anno if a["evaluation_id"] == row["evaluation_id"]]
        max_diff = max((abs(c["score"] - j["score"])
                        for a in r for c in [dict(criterion=k, score=v) for k,v in json.loads(a["scores"])["scores"].items()]
                        for j in judge["criteria"] if j["criterion"] == c["criterion"] and c["score"] is not None and j["score"] is not None), default=-1)
        scored.append((row, max_diff))
    minimum = st.slider("Minimum criterion disagreement", -1, 4, 1)
    candidates = [r for r, d in scored if d >= minimum]
    if not candidates:
        st.info("No cases match selected difference threshold.")
        st.stop()
    selected = st.selectbox("Case", [r["evaluation_id"] for r in candidates])
    render_item(store.item(selected))
    result = next(x for x in candidates if x["evaluation_id"] == selected)
    st.write("Judge per-criterion judgments")
    st.dataframe(pd.DataFrame(json.loads(result["payload"])["criteria"]), hide_index=True)
    ratings = [r for r in anno if r["evaluation_id"] == selected]
    st.write("Independent submitted human ratings")
    st.dataframe(pd.DataFrame([{"annotator": r["annotator_id"], "revision": r["revision"],
                               **json.loads(r["scores"])} for r in ratings]), hide_index=True)
    st.json([a for a in store.adjudications() if a["evaluation_id"] == selected and a["rubric_key"] == current_rubric])
    if len(ratings) >= 2:
        rubric = store.rubric(current_rubric)
        with st.form("adjudicate"):
            adjudicator = st.text_input("Adjudicator pseudonym")
            decision = {}
            for c in rubric.criteria:
                if c.enabled:
                    option = st.selectbox(c.name, ["Cannot assess"] + [f"{k}: {v}" for k, v in sorted(c.anchors.items())], key=f"adj_{c.key}")
                    decision[c.key] = None if option == "Cannot assess" else int(option.split(":")[0])
            reason = st.text_area("Rationale and evidence used to reconcile ratings")
            if st.form_submit_button("Save adjudication"):
                try:
                    store.adjudicate(selected, adjudicator, HumanScores(scores=decision), reason, rubric_key=current_rubric)
                    st.success("Adjudication recorded separately; individual ratings preserved.")
                except Exception as exc:
                    message_error(exc)
    else:
        st.info("Adjudication requires two independent submitted human annotations.")

elif page == "Bias & Robustness":
    st.subheader("Controlled bias, injection and repeatability experiments")
    experiments = store.experiments()
    if experiments:
        st.dataframe(pd.DataFrame([{"id": e["experiment_id"], "type": e["type"],
                                    "item": e["evaluation_id"], "time": e["created_at"]} for e in experiments]), hide_index=True)
        picked = st.selectbox("Inspect experiment", [e["experiment_id"] for e in experiments])
        st.json(json.loads(next(e["payload"] for e in experiments if e["experiment_id"] == picked)))
    else:
        st.info("No robustness experiments have run. No outcomes are inferred.")
    items = store.items()
    if items:
        with st.form("experiment"):
            chosen = st.selectbox("Evaluation record", [i.evaluation_id for i in items])
            kind = st.selectbox("Experiment", ["verbosity", "presentation", "injection", "attribution", "reference", "repeat"])
            variant = st.text_area("Controlled variant answer (required except reference/repeat)")
            count = st.number_input("Repetitions per condition", 1, 20, 2)
            if st.form_submit_button("Run paid experiment"):
                try:
                    cfg = JudgeConfig.model_validate(get_config())
                    judge = ClaudeJudge(cfg)
                    payload = paired_experiment(store, judge, chosen, kind, variant or None, count)
                    st.json(payload["summary"])
                except Exception as exc:
                    message_error(exc)
    runs = run_options()
    if runs:
        selected = st.selectbox("Inspect repeat stability from a persisted run", runs)
        report = stability(store, selected)
        if report:
            st.json(report)
        else:
            st.info("Repeat counts below 2 cannot estimate rating stability.")
    st.caption("Verbosity experiments require equivalent factual content; attribution needs source-controlled items. A paired result alone is not proof of a causal bias.")

elif page == "Reports & Export":
    st.subheader("Research report from persisted evidence")
    runs = run_options()
    must_have(runs, "evaluation runs")
    run_id = st.selectbox("Evaluation run", runs)
    strategy = st.selectbox("Human-score strategy", ["adjudicated", "single", "majority", "mean"])
    fmt = st.selectbox("Output format", ["markdown", "json", "csv", "html"])
    report = report_dict(store, run_id, strategy, n_boot=300)
    data = render_report(report, fmt)
    st.download_button("Download generated report", data, file_name=f"judgebench_report.{ 'md' if fmt == 'markdown' else fmt}")
    st.write("Report composition and usage (persisted data only):")
    st.json({"dataset": report["sample"], "usage": report["usage"], "method": report["method"]})
    st.caption("Reports do not present human labels or empirical agreement from test fixtures.")
