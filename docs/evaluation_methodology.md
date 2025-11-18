# Evaluation study methodology

1. Define a target population and prospective sampling frame. Include answer errors, correct answers, short and long responses, categories, difficulty and generator provenance. Separate development and final holdout *before* prompt iteration.
2. Freeze a rubric and prompt version; pilot on development only. If clarification is necessary, version the rubric rather than changing historical anchors.
3. Run Claude with permitted original tasks, candidate responses, references and evidence. `models.retrieve` verifies the chosen provider model on each run. Structured output ensures valid shape only; semantic criteria undergo additional checks.
4. Collect independent blind human labels with comments, revisions and adjudication. At least two raters on a stratified overlap allow measuring human–human consistency.
5. Run preplanned criterion-level agreement, signed bias, confusion matrices and uncertainty. Inspect disagreements and missed factual/grounding failures manually, especially cases with low human agreement. Report abstention coverage and provider failures separately.
6. Compare v1/v2 prompts or rubric changes using `compare-runs` on explicitly `holdout`-marked, item-matched tasks. For cross-rubric agreement differences the criterion definitions must be identical and both human references must be present and agree. Report unavailable comparisons rather than selecting favorable cases after inspecting results.
7. Repeat experiments with controlled equivalence and keep raw responses/rationales for manual review. Automated regressions do not establish generalization beyond the observed collection.

**Research manuscript outline:** Abstract, Introduction, Related Work, Dataset Collection and Provenance, Human Annotation and Calibration, Rubric Design, Claude Configuration, Agreement Statistics and Uncertainty, Robustness Experiments, Disagreement Case Studies, Discussion, Limitations, Ethics, References (only verified works). Do not fabricate sample sizes, confidence intervals or reference citations.
