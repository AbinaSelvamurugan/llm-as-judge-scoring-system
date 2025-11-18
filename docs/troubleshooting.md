# Troubleshooting

- `No module named judgebench`: install in editable mode (`pip install -e .[dev]`) or set `PYTHONPATH=src` when running `python -m judgebench.cli`.
- `ANTHROPIC_API_KEY is not configured`: export it in the *same shell*. `.env` is a template, not an auto-loader.
- `offline mode blocks all live API calls`: remove `JUDGEBENCH_OFFLINE=1` only after authorization for the live run.
- Model unavailable: confirm account model access with Anthropic API credentials and `client.models.retrieve`; try an accessible supported model ID. Provider errors must not be mistaken for grades.
- Incompatible structured outputs: choose `--unstructured-fallback` for explicit bounded JSON parsing (may be less robust) and verify outputs; unsupported model should be switched to a documented supported one where possible.
- `unknown rubric`: run `judgebench init` or `judgebench rubric-add configs/rubric.yaml` before importing records.
- Duplicate import: IDs and normalized question-answer text are unique; inspect and resolve duplicates rather than overriding.
- No eligible agreement: submit human ratings for the **same item and rubric version**, run the provider successfully, select `single/majority/adjudicated` appropriate to the data. Null/abstained scores do not count as paired grades.
- Streamlit not installed: use the pinned requirements on a network-enabled machine; `python -m compileall` checks only syntax, not widget behavior.
- No pricing: token counts may be available; monetary expense intentionally remains unavailable without verified token prices.
