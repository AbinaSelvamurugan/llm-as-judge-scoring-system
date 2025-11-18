# Human annotation and adjudication protocol

1. Obtain authorized, non-sensitive task-answer pairs and separate them into development versus untouched holdout before tuning.
2. Recruit and train raters on **criterion-specific anchors**, boundaries, meaningful abstentions and how to check supplied evidence. Pilot only on development tasks. Document disagreement patterns and clarify a *new rubric version* rather than silently editing one.
3. Assign blinded IDs and sample two independent raters for a stratified shared subset (for example, across topics and apparent answer quality). Single-rate remaining examples only where justified by the study design; do not assume all human judgments are objective truth.
4. Expose original prompt, candidate answer, references and rubric instructions only, never Claude results or prior annotator decisions. Use the `Human Annotation` view with annotator credentials. Save progress in drafts, submit each enabled criterion (or `Cannot assess`), and leave optional evidence-annotated comments.
5. Revisions are immutable, timestamped and linked to previous annotation IDs. Latest submitted revision per rater is the default for analysis. Adjudicators may view all independent human ratings *after submission*, review the evidence, document a reason and record a separate adjudicated result without deleting raw judgments.
6. Export research results with pseudonymous IDs; keep the annotator-identity mapping outside the database. Auditors should review selected human–judge disagreements and any finding where evidence quality is deficient.

**Blindness qualification:** Unless both access tokens are configured and reviewer views are protected by a trusted upstream authenticated service, the dashboard is suitable only for trusted local use, not high-stakes blinded human research. A password shared among annotators cannot verify identity or enforce individual task allocation. Never infer labels from the synthetic fixture.
