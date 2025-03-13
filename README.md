# JudgeBench — Rubric-Based Evaluation of LLM Answers Against Human Judgments

JudgeBench evaluates task/answer pairs with explicit, criterion-specific score anchors. It calls the Claude Messages API for schema-validated grading, collects independent human labels, retains adjudication and revision history, and reports human–judge agreement with sample counts, missingness and uncertainty.

It answers a set of empirical questions: how often judge and humans agree, which criteria differ, whether ratings run generous or harsh, whether answer length or presentation shifts scores, whether correctness and relevance get conflated, whether the judge misses unsupported claims, whether repeat judgments are stable, how results vary by task category, whether a changed prompt helps on untouched holdouts, and which decisions need human review.

The project is released under the MIT License (`LICENSE`), with contribution guidance in `CONTRIBUTING.md` and an offline test workflow at `.github/workflows/tests.yml`. No secret tokens, live SQLite data, or study results are included.

## Model configuration

The SDK API implementation targets Anthropic Python SDK 1.13.0 and `client.messages.parse(output_format=ProviderJudgment)` (the SDK sends `output_config.format`), with an opt-in, schema-validated `messages.create` JSON fallback. See [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) and [model deprecations](https://docs.anthropic.com/en/docs/about-claude/model-deprecations).

The default model identifier is `claude-sonnet-4-6`. At run time the application calls `client.models.retrieve(model)` to confirm the model is available to the account.

## Architecture

```text
CSV/JSONL -> validated EvaluationItem -> SQLite (immutable items + rubric versions)
                                                  |
                 run snapshot (item IDs + config + prompt snapshot)
                                                  |
                   Claude Messages API -> schema + rubric validation
                                                  |
                                      saved judgments / failures
                                                  |
                  independent blind human annotations / revisions
                                                  |
                      adjudication without overwriting raters
                                                  |
                   agreement + bootstrap + robustness + reports
                                                  |
                                CLI and 10-view Streamlit UI
```

### Installation (from repository root)

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env                  # copy values manually; .env is NOT auto-loaded
export ANTHROPIC_API_KEY='your-key'  # PowerShell: $env:ANTHROPIC_API_KEY='your-key'
export JUDGEBENCH_DB=judgebench.sqlite3
```

Dependencies are version-pinned in `pyproject.toml`, and `requirements.txt` installs them together with the development tools. Offline core tests run against the installed statistical libraries:

```bash
PYTHONPATH=src python -m pytest -q
```

### Getting started

```bash
judgebench init
judgebench rubric-check --rubric configs/rubric.yaml
judgebench dataset-validate data/synthetic_fixture.jsonl
judgebench dataset-import data/synthetic_fixture.jsonl
judgebench dataset-export --version synthetic-test-v1 --format csv --out results/items.csv
judgebench run --dataset synthetic-test-v1 --rubric-key default@1.0.0 --limit 1
judgebench list-runs
judgebench agreement --run RUN_UUID --strategy single --bootstrap 500
judgebench human-agreement --rubric-key default@1.0.0
judgebench stability --run RUN_UUID
judgebench report --run RUN_UUID --strategy adjudicated --format markdown --out results/report.md
streamlit run app/streamlit_app.py
```

The `run` command issues a paid provider request when credentials and access are available. For substantive research, point it at a documented, approved, diverse dataset. To keep a session fully offline:

```bash
export JUDGEBENCH_OFFLINE=1
pytest -q
# Any attempted JudgeBench API evaluation is rejected in offline mode.
```

A benchmark study is built from model-generated answers and blind human annotations:

```bash
judgebench dataset-validate data/my_evaluations.jsonl
judgebench dataset-import data/my_evaluations.jsonl
judgebench run --dataset my-study-v1 --rubric-key default@1.0.0 --limit 250 --batch-size 5 --repeats 1
# record the UUID emitted by the command; in a separate shell you may request cancellation:
judgebench cancel RUN_UUID
judgebench resume RUN_UUID
```

Judge runs use sequential bounded batches, not simultaneous API requests. Retries for API timeouts, transient failures and rate limits are bounded. A canceled run stops between requests; it cannot terminate an already in-flight API request. Completed `(run_id, evaluation_id, repeat_index)` records are not rerun by resume. Failed ones may be retried.

### Rubrics and score interpretation

`configs/rubric.yaml` provides complete definitions, instructions, anchors, examples, evidence requirements, critical thresholds, weights and abstention rules for seven criteria:

| Criterion | Default | Scale | Special policy |
|---|---|---|---|
| Factual correctness | Enabled | 1–5 | <=2 critical failure |
| Relevance | Enabled | 1–5 | Distinct from truthfulness |
| Completeness | Enabled | 1–5 | Doesn't equate length with coverage |
| Grounding | Enabled | 1–5 | Must abstain with no supplied evidence; <=2 critical |
| Instruction following | Enabled | 1–5 | Includes scope and output constraints |
| Clarity | Disabled (optional) | 1–5 | Weight zero by default |
| Safety | Disabled (optional) | 0–1 | Needs applicable safety policy, 0 is critical |

Only enabled criteria are scored. Ordinal values map to `(score-1)/4` and binary values map to `0/1`; a weighted normalized average is calculated only over assessed enabled criteria. The default pass threshold is 0.65. Critical failures override passing averages; if no critical failure occurs but a criterion abstains, the thresholded decision is `not_assessable`. Missing results are never silently assigned zero, and missing expected criterion fields make a provider judgment invalid. Use rubric-specific criteria and explicit critical rules for specialized tasks. See [rubric design](docs/rubric_design.md).

Rubrics are immutable after insertion; updating weights, anchors, scales or examples requires a new `rubric_id@version`. The same versioned dataset may be judged under multiple rubric versions. Human annotators must separately select the rubric version; cross-rubric comparisons require meaningful common anchors and separate human ratings. Prompt templates `v1` and `v2` are snapshot-per-run. Use `judgebench rubric-add path/to/another.yaml` or the Rubric Builder UI.

### Dataset schema

Required JSONL/CSV fields: `evaluation_id`, `question`, `answer`, `dataset_version`, `rubric_key`. Optional: `reference_answer`, `evidence`, `category`, `split` (`development`, `holdout`, `unspecified`), `difficulty`, `provenance`, `generator_model`. `rubric_key` in the item is the default annotation rubric, but the run selects its own explicit rubric. Every task/answer pair has a unique ID and a deduplicated normalized content fingerprint. Import checks entire CSV/JSONL records, enforces 8 MB and 20,000 item limits, and fails transactionally on duplicates. `run_items` freezes the exact item set at run creation; later imports cannot change an existing run's denominator.

The bundled `data/synthetic_fixture.jsonl` is a small synthetic fixture for schema and integration checks; it holds no human judgments and is not a benchmark.

Human labels are not accepted in dataset imports: they are recorded separately as authenticated or pseudonymous reviewer actions with timestamps and revision chains. This keeps independent annotations from being mistaken for data fields. A study item looks like this:

```json
{"evaluation_id":"task-0001","question":"What did the provided report conclude?","answer":"...","reference_answer":"...","evidence":"...","category":"summarization","split":"holdout","dataset_version":"my-study-v1","rubric_key":"default@1.0.0","provenance":"consented study collection"}
```

### Independent human annotation and adjudication

Launch Streamlit, go to Human Annotation, enter an assigned pseudonymous identifier, select the rubric/version, task category and task. Rate each enabled criterion from the displayed anchors, or choose Cannot assess when allowed. Drafts save partial progress; submission requires all criterion fields. Revising a submitted annotation creates a new immutable revision and audit link. Claude grades are never displayed in the annotation page. In multi-user studies, configure separate annotation and reviewer access tokens and isolate the reviewer pages so unsubmitted annotations cannot be contaminated:

```bash
export JUDGEBENCH_ANNOTATOR_TOKEN='separate-long-random-token'
export JUDGEBENCH_REVIEW_TOKEN='another-long-random-token'
streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Without both tokens, the dashboard runs in a labelled local-only mode with no access protection. Those tokens are lightweight application gates, not a substitute for authenticated deployment, TLS, reverse-proxy permissions, per-user identities or study governance. Annotator IDs are self-asserted pseudonyms; assignment and identity verification are external study responsibilities. Reviewers can inspect disagreements and adjudicate only after at least two raters have independently submitted. Raw ratings remain stored after adjudication.

CLI annotations use a JSON map for all enabled scores:

```bash
judgebench annotate task-0001 --annotator rater-0003 --rubric-key default@1.0.0 \
  --scores-json '{"factual":5,"relevance":5,"completeness":4,"grounding":4,"instruction":5}' \
  --comments 'Reference checked.'
judgebench adjudicate task-0001 --adjudicator senior-1 --rubric-key default@1.0.0 \
  --scores-json '{"factual":4,"relevance":5,"completeness":4,"grounding":4,"instruction":5}' \
  --reason 'Adjudicated against documentary evidence after independent ratings.'
```

`null` is the Cannot assess score. The annotation page never shows the LLM grade, even after submission; only access-controlled reviewer pages may show it.

### Agreement and experiments

Agreement is per criterion and version: exact agreement, ordinal within-N tolerance (default 1), raw confusion matrix (human rows, Claude columns), unweighted and quadratic-weighted Cohen's kappa, Spearman/Kendall, signed error and MAE, and binary precision/recall/F1 for the positive label `1`. Human–human agreement covers co-rated item pairs; the bootstrap is clustered by evaluation item. Valid pairs exclude null/missing entries, and denominators and score distributions are always shown. Kappa is undefined for certain degenerate label patterns; the software reports `null` rather than fabricating a number. Reference strategies are `single` (first pseudonym lexicographically), `majority` (ties abstain), `adjudicated` (requires adjudication) and `mean` (MAE/bias only, because a fractional human mean cannot be used as an ordinal category). See [agreement metrics](docs/agreement_metrics.md).

```bash
judgebench agreement --run RUN_UUID --strategy adjudicated --bootstrap 1000
judgebench compare-runs --a RUN_UUID_A --b RUN_UUID_B --strategy adjudicated
judgebench experiment --item task-0001 --kind verbosity --variant-text 'Alternative equally correct answer' --runs 3
judgebench experiment --item task-0001 --kind injection --variant-text 'An answer containing an embedded grader instruction' --runs 3
judgebench experiment --item task-0001 --kind reference --runs 3
judgebench experiment --item task-0001 --kind repeat --runs 5
```

`compare-runs` requires the same dataset version, shared records marked `split="holdout"`, identical criterion definitions, and matching human reference judgments. It calculates paired changes in exact agreement; no comparisons are reported where these controls are unmet. Robustness comparisons call the API; controlled text variants are supplied by the researcher, who must ensure substantive equivalence and independently document the manipulation. Attribution tests also require controlled generator provenance; the software never infers bias from non-equivalent outputs. Repeatability tracks distinct scores and literal rationale changes. Paid experiment outputs are stored.

### Reports and development commands

```bash
judgebench report --run RUN_UUID --strategy single --format json --out results/report.json
judgebench report --run RUN_UUID --format csv --out results/criteria.csv
judgebench report --run RUN_UUID --format html --out results/report.html
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src app scripts
```

Report formats are CSV, JSON, Markdown and standalone HTML (HTML currently uses escaped plain-text formatting). They show run metadata, counts, usage, category breakdown, per-criterion agreement, confusion matrices, human–human ratings, error classes and explicit limitations. Monetary cost is reported only when provider pricing is supplied and verified. Item-level disagreements can be explored within Streamlit.

### Data privacy and deployment

Never commit `.env`, credentials or a live SQLite database. API inputs include original questions, candidate answers, rubric instructions, and any supplied reference/evidence. They leave the machine and are sent to Anthropic when a live evaluation runs. An annotation-only session does not make LLM requests. The application does not automatically anonymize sensitive datasets; use only authorized material. Do not host the local SQLite database on an exposed public interface. See [security](docs/security.md), [deployment](docs/deployment.md) and [annotation protocol](docs/annotation_protocol.md).

## Reproducibility and known limitations

- Model aliases and non-deterministic inference prevent identical reruns. Exact model ID, returned provider model, token counts, prompt snapshot and rubric definition are persisted; sampling settings remain provider defaults and may change.
- Two prompts and flexible rubric versioning exist, but strong conclusions require separate development and untouched holdout data and a pre-registered comparison plan. Do not tune against holdout examples.
- Some study requirements are outside the local tooling: permission-based assignment, encrypted data storage, enterprise authentication, external storage, multi-process orchestration and in-flight request cancellation.
- Structural JSON correctness does not imply grading accuracy. Reliability evidence comes from independent, experienced human annotation, adjudication and error auditing.

See all documentation under `docs/`.
