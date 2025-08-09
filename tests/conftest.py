import pytest
from judgebench.rubrics import load_rubric
from judgebench.storage import Store
from judgebench.datasets import load_path


@pytest.fixture
def rubric():
    return load_rubric("configs/rubric.yaml")


@pytest.fixture
def store(tmp_path, rubric):
    s = Store(tmp_path / "db.sqlite3")
    s.add_rubric(rubric)
    s.add_items(load_path("data/synthetic_fixture.jsonl"))
    return s
