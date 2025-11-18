# Agreement metrics and uncertainty

For each criterion and exactly matched `(evaluation_id, rubric_key)`:

- **Eligible pairs:** one completed judge sample (`repeat_index=0`) and a non-null selected human rating. Missing and `Cannot assess` are excluded but their counts are reported.
- **Exact:** number of equal ratings / eligible count.
- **Within N:** ordinal only, |Claude−human| ≤ N. `N=1` default.
- **Confusion matrix:** ordered rubric labels, human rows, Claude columns, raw counts.
- **Cohen's kappa:** unweighted agreement beyond chance; **quadratic weighted kappa** penalizes larger ordinal distances more. Kappa can be undefined in degenerate conditions and unstable for imbalanced labels.
- **Rank association:** Spearman and Kendall detect ordering similarity, not absolute grade correctness.
- **Error:** MAE = mean |Claude−human|, signed bias = mean (Claude−human). Higher positive bias indicates generous scoring.
- **Binary:** precision, recall and F1 with `1` as positive class and explicitly recorded `0` as negative.
- **Human–human:** all pairwise independent annotator pairs on co-rated items, clustering bootstrap resamples over evaluation items (not each paired rating independently). Multirater reliability can additionally require preplanned Krippendorff alpha; this implementation does not claim to estimate it.
- **Uncertainty:** percentile nonparametric bootstrap over unique evaluation IDs with fixed seed, default 500 replicates and 95% CI. A CI of `null` indicates too little information, insufficient resamples or undefined statistic. Bootstrap CIs for item-level metrics are descriptive; no statistical significance testing is asserted.

`single` is an arbitrary but reproducible first annotator by ID; `majority` resolves only unambiguous wins, with ties as abstentions; `adjudicated` selects the latest documented adjudication, and `mean` supports **only numeric signed-bias/MAE**, not ordinal kappa/confusion matrices. An adjudicated rating does not magically become objective truth.

**Study reporting:** include eligible and total case counts, missingness, label prevalence/distributions, human selection method, confidence intervals, category breakdowns, repeated-measure policy and representative audited disagreements. Stratify by difficulty and answer-quality where annotated. Do not describe a kappa threshold as proof of being 'human-level'. When comparing prompts, use item-matched heldout examples only and avoid treating rubric definitions with changed meanings as directly comparable.
