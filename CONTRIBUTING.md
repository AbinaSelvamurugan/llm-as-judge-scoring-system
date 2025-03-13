# Contributing

Use Python 3.11–3.14 and pinned dependencies. Run `pytest -q`, `ruff check .` and `ruff format --check .` before proposing changes. Keep SDK integration tests with real credentials separate from offline tests; never introduce a fake scorer into production. All new rubric versions are immutable. Preserve item IDs, annotation revisions, missing/abstained values and heldout splits. Do not commit real human labels or datasets without documented permission. New metrics must include deterministic unit tests with known expected values and explicit denominators.
