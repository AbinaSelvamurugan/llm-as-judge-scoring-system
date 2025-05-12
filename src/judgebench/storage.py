"""Transactional SQLite persistence with immutable rubric snapshots and human revision history."""
from __future__ import annotations
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from .schemas import EvaluationItem, Rubric, HumanScores, JudgeConfig, utcnow
from .rubrics import rubric_json, validate_scores


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or os.getenv("JUDGEBENCH_DB", "judgebench.sqlite3"))
        self.initialize()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=15000")
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS rubrics (
                  rubric_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS items (
                  evaluation_id TEXT PRIMARY KEY, dataset_version TEXT NOT NULL,
                  rubric_key TEXT NOT NULL REFERENCES rubrics(rubric_key), payload TEXT NOT NULL,
                  fingerprint TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, config TEXT NOT NULL, rubric_key TEXT NOT NULL,
                  dataset_version TEXT NOT NULL, prompt_template TEXT NOT NULL,
                  status TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT);
                CREATE TABLE IF NOT EXISTS run_items (
                  run_id TEXT NOT NULL REFERENCES runs(run_id),
                  evaluation_id TEXT NOT NULL REFERENCES items(evaluation_id),
                  position INTEGER NOT NULL,
                  PRIMARY KEY(run_id,evaluation_id));
                CREATE TABLE IF NOT EXISTS judgments (
                  run_id TEXT NOT NULL REFERENCES runs(run_id),
                  evaluation_id TEXT NOT NULL REFERENCES items(evaluation_id),
                  repeat_index INTEGER NOT NULL,
                  state TEXT NOT NULL, payload TEXT, aggregate TEXT,
                  error_type TEXT, error_message TEXT, tokens_in INTEGER, tokens_out INTEGER,
                  latency_seconds REAL, provider_model TEXT,
                  completed_at TEXT NOT NULL,
                  PRIMARY KEY (run_id,evaluation_id,repeat_index));
                CREATE TABLE IF NOT EXISTS annotations (
                  annotation_id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES items(evaluation_id),
                  rubric_key TEXT NOT NULL, annotator_id TEXT NOT NULL,
                  revision INTEGER NOT NULL, state TEXT NOT NULL, scores TEXT NOT NULL,
                  created_at TEXT NOT NULL, previous_annotation_id TEXT,
                  UNIQUE (evaluation_id,rubric_key,annotator_id,revision));
                CREATE TABLE IF NOT EXISTS adjudications (
                  adjudication_id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES items(evaluation_id),
                  rubric_key TEXT NOT NULL, adjudicator_id TEXT NOT NULL,
                  scores TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments (
                  experiment_id TEXT PRIMARY KEY, type TEXT NOT NULL, run_id TEXT,
                  evaluation_id TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    def add_rubric(self, rubric: Rubric):
        payload = rubric_json(rubric)
        with self.connection() as db:
            row = db.execute("SELECT payload FROM rubrics WHERE rubric_key=?", (rubric.key,)).fetchone()
            if row and row["payload"] != payload:
                raise ValueError("rubric versions are immutable; choose a new version")
            db.execute("INSERT OR IGNORE INTO rubrics VALUES (?,?,?)", (rubric.key, payload, utcnow()))

    def rubrics(self) -> list[Rubric]:
        with self.connection() as db:
            return [Rubric.model_validate_json(r["payload"]) for r in db.execute("SELECT payload FROM rubrics ORDER BY rubric_key")]

    def rubric(self, key: str) -> Rubric:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM rubrics WHERE rubric_key=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"unknown rubric version: {key}")
        return Rubric.model_validate_json(row["payload"])

    def add_items(self, items: list[EvaluationItem]) -> int:
        from .datasets import _digest
        if len({x.evaluation_id for x in items}) != len(items):
            raise ValueError("duplicate evaluation IDs in batch")
        with self.connection() as db:
            existing = {r["fingerprint"] for r in db.execute("SELECT fingerprint FROM items")}
            count = 0
            for item in items:
                self.rubric(item.rubric_key)
                fingerprint = _digest(item)
                if fingerprint in existing:
                    # Duplicate content is a different item only if explicitly curated elsewhere.
                    raise ValueError(f"duplicate question-answer pair: {item.evaluation_id}")
                db.execute("INSERT INTO items VALUES (?,?,?,?,?,?)",
                           (item.evaluation_id, item.dataset_version, item.rubric_key,
                            item.model_dump_json(), fingerprint, utcnow()))
                existing.add(fingerprint)
                count += 1
            return count

    def items(self, dataset_version: str | None = None, rubric_key: str | None = None) -> list[EvaluationItem]:
        sql = "SELECT payload FROM items WHERE 1=1"
        args = []
        if dataset_version:
            sql += " AND dataset_version=?"
            args.append(dataset_version)
        if rubric_key:
            sql += " AND rubric_key=?"
            args.append(rubric_key)
        with self.connection() as db:
            return [EvaluationItem.model_validate_json(r["payload"]) for r in db.execute(sql + " ORDER BY evaluation_id", args)]

    def item(self, evaluation_id: str) -> EvaluationItem:
        with self.connection() as db:
            r = db.execute("SELECT payload FROM items WHERE evaluation_id=?", (evaluation_id,)).fetchone()
        if r is None:
            raise ValueError("unknown evaluation ID")
        return EvaluationItem.model_validate_json(r["payload"])

    def create_run(self, config: JudgeConfig, rubric: Rubric, dataset_version: str, prompt_template: str) -> str:
        self.add_rubric(rubric)
        run_id = str(uuid.uuid4())
        captured = self.items(dataset_version)[:config.limit]
        if not captured:
            raise ValueError("run requires at least one matching dataset item")
        with self.connection() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                       (run_id, config.model_dump_json(), rubric.key, dataset_version,
                        prompt_template, "pending", utcnow(), None))
            db.executemany("INSERT INTO run_items VALUES (?,?,?)",
                           [(run_id, x.evaluation_id, i) for i, x in enumerate(captured)])
        return run_id

    def run_items(self, run_id: str) -> list[EvaluationItem]:
        with self.connection() as db:
            rows = db.execute("""SELECT i.payload FROM run_items ri
                   JOIN items i ON ri.evaluation_id=i.evaluation_id
                   WHERE ri.run_id=? ORDER BY ri.position""", (run_id,)).fetchall()
        return [EvaluationItem.model_validate_json(r["payload"]) for r in rows]

    def runs(self) -> list[dict]:
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY created_at DESC")]

    def run(self, run_id: str) -> dict:
        with self.connection() as db:
            r = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if r is None:
            raise ValueError("unknown run")
        return dict(r)

    def set_run_state(self, run_id: str, state: str):
        if state not in {"pending", "running", "completed", "failed", "cancelled", "partial"}:
            raise ValueError("invalid run state")
        with self.connection() as db:
            db.execute("UPDATE runs SET status=?,completed_at=? WHERE run_id=?",
                       (state, utcnow() if state in {"completed", "failed", "cancelled", "partial"} else None, run_id))

    def save_result(self, run_id: str, evaluation_id: str, repeat: int, *, judgment=None,
                    aggregate=None, error_type=None, error_message=None, tokens_in=None,
                    tokens_out=None, latency_seconds=None, provider_model=None):
        state = "completed" if judgment is not None else "failed"
        with self.connection() as db:
            db.execute("""INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(run_id,evaluation_id,repeat_index) DO UPDATE SET
                       state=excluded.state,payload=excluded.payload,aggregate=excluded.aggregate,
                       error_type=excluded.error_type,error_message=excluded.error_message,
                       tokens_in=excluded.tokens_in,tokens_out=excluded.tokens_out,
                       latency_seconds=excluded.latency_seconds,provider_model=excluded.provider_model,
                       completed_at=excluded.completed_at""",
                       (run_id, evaluation_id, repeat, state,
                        judgment.model_dump_json() if judgment else None,
                        aggregate.model_dump_json() if aggregate else None,
                        error_type, (error_message or "")[:300], tokens_in, tokens_out,
                        latency_seconds, provider_model, utcnow()))

    def results(self, run_id: str | None = None) -> list[dict]:
        with self.connection() as db:
            query = "SELECT * FROM judgments"
            params = ()
            if run_id:
                query += " WHERE run_id=?"
                params = (run_id,)
            return [dict(r) for r in db.execute(query + " ORDER BY run_id,evaluation_id,repeat_index", params)]

    def annotate(self, evaluation_id: str, annotator_id: str, human: HumanScores,
                 state: str = "submitted", rubric_key: str | None = None) -> str:
        if state not in {"draft", "submitted"}:
            raise ValueError("invalid annotation state")
        if not annotator_id or len(annotator_id) > 100:
            raise ValueError("pseudonymous annotator ID required")
        item = self.item(evaluation_id)
        rubric = self.rubric(rubric_key or item.rubric_key)
        # Draft annotations may be incomplete, submissions require all criteria.
        if state == "submitted":
            validate_scores(rubric, human.scores)
        else:
            active = {c.key for c in rubric.criteria if c.enabled}
            if not set(human.scores) <= active:
                raise ValueError("unknown draft criterion")
            for key, value in human.scores.items():
                criterion = next(c for c in rubric.criteria if c.key == key)
                if value is not None and value not in criterion.anchors:
                    raise ValueError("invalid draft score")
        prior = self.latest_annotations(evaluation_id=evaluation_id, annotator_id=annotator_id, rubric_key=rubric.key)
        last = prior[0] if prior else None
        revision = (last["revision"] + 1) if last else 1
        if last and last["state"] == "submitted" and state != "submitted":
            raise ValueError("submitted annotation can only be changed by a new submitted revision")
        annotation_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?,?)",
                       (annotation_id, evaluation_id, rubric.key, annotator_id,
                        revision, state, human.model_dump_json(), utcnow(),
                        last["annotation_id"] if last else None))
        return annotation_id

    def latest_annotations(self, evaluation_id: str | None = None,
                           annotator_id: str | None = None, submitted_only=False,
                           rubric_key: str | None = None) -> list[dict]:
        query = """SELECT a.* FROM annotations a JOIN
                   (SELECT evaluation_id,rubric_key,annotator_id,MAX(revision) rev
                    FROM annotations GROUP BY evaluation_id,rubric_key,annotator_id) last
                   ON a.evaluation_id=last.evaluation_id AND a.rubric_key=last.rubric_key
                   AND a.annotator_id=last.annotator_id AND a.revision=last.rev WHERE 1=1"""
        args = []
        if evaluation_id:
            query += " AND a.evaluation_id=?"
            args.append(evaluation_id)
        if annotator_id:
            query += " AND a.annotator_id=?"
            args.append(annotator_id)
        if rubric_key:
            query += " AND a.rubric_key=?"
            args.append(rubric_key)
        if submitted_only:
            query += " AND a.state='submitted'"
        with self.connection() as db:
            return [dict(r) for r in db.execute(query + " ORDER BY a.evaluation_id,a.annotator_id", args)]

    def annotation_history(self, evaluation_id: str, annotator_id: str) -> list[dict]:
        with self.connection() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM annotations WHERE evaluation_id=? AND annotator_id=? ORDER BY revision",
                (evaluation_id, annotator_id))]

    def adjudicate(self, evaluation_id: str, adjudicator_id: str, scores: HumanScores, reason: str,
                   rubric_key: str | None = None):
        if not reason.strip():
            raise ValueError("adjudication requires rationale")
        item = self.item(evaluation_id)
        rubric_key = rubric_key or item.rubric_key
        validate_scores(self.rubric(rubric_key), scores.scores)
        if len(self.latest_annotations(evaluation_id, submitted_only=True, rubric_key=rubric_key)) < 2:
            raise ValueError("adjudication requires at least two submitted independent raters")
        with self.connection() as db:
            db.execute("INSERT INTO adjudications VALUES (?,?,?,?,?,?,?)",
                       (str(uuid.uuid4()), evaluation_id, rubric_key, adjudicator_id,
                        scores.model_dump_json(), reason, utcnow()))

    def adjudications(self) -> list[dict]:
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM adjudications ORDER BY created_at")]

    def add_experiment(self, experiment_type: str, payload: dict, run_id=None, evaluation_id=None):
        with self.connection() as db:
            db.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", (str(uuid.uuid4()),
                       experiment_type, run_id, evaluation_id, json.dumps(payload), utcnow()))

    def experiments(self) -> list[dict]:
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM experiments ORDER BY created_at DESC")]

    def summary(self) -> dict:
        with self.connection() as db:
            scalar = lambda sql: db.execute(sql).fetchone()[0]
            return {"items": scalar("SELECT COUNT(*) FROM items"),
                    "judgments": scalar("SELECT COUNT(*) FROM judgments WHERE state='completed'"),
                    "api_failures": scalar("SELECT COUNT(*) FROM judgments WHERE state='failed'"),
                    "submitted_human": scalar("SELECT COUNT(*) FROM annotations WHERE state='submitted'"),
                    "annotated_items": scalar("SELECT COUNT(DISTINCT evaluation_id) FROM annotations WHERE state='submitted'"),
                    "runs": scalar("SELECT COUNT(*) FROM runs")}
