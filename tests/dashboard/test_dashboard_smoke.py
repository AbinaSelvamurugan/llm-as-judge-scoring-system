"""Executed only where Streamlit is installed, otherwise explicitly skipped."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "app" / "streamlit_app.py"


def test_dashboard_empty_overview_runs_without_crashing(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("JUDGEBENCH_DB", str(tmp_path / "ui.sqlite3"))
    monkeypatch.delenv("JUDGEBENCH_ANNOTATOR_TOKEN", raising=False)
    monkeypatch.delenv("JUDGEBENCH_REVIEW_TOKEN", raising=False)
    test = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not test.exception
    assert any("JudgeBench" in x.value for x in test.title)


def test_dashboard_rubric_builder_navigation(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from judgebench.storage import Store
    from judgebench.rubrics import load_rubric

    path = str(tmp_path / "ui.sqlite3")
    db = Store(path)
    db.add_rubric(load_rubric(str(REPO_ROOT / "configs" / "rubric.yaml")))
    monkeypatch.setenv("JUDGEBENCH_DB", path)
    monkeypatch.delenv("JUDGEBENCH_ANNOTATOR_TOKEN", raising=False)
    monkeypatch.delenv("JUDGEBENCH_REVIEW_TOKEN", raising=False)
    test = AppTest.from_file(str(APP), default_timeout=20).run()
    test.sidebar.radio[0].set_value("Rubric Builder").run()
    assert not test.exception
