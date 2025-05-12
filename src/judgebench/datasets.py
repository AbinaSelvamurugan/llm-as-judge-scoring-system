"""Validated bounded CSV/JSONL import/export and duplication control."""
import csv
import hashlib
import io
import json
from pathlib import Path
from pydantic import ValidationError
from .schemas import EvaluationItem

MAX_FILE_BYTES = 8_000_000


def _digest(item: EvaluationItem) -> str:
    return hashlib.sha256((item.question.strip().casefold() + "\x00" + item.answer.strip().casefold()).encode()).hexdigest()


def load_bytes(data: bytes, suffix: str) -> list[EvaluationItem]:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("file exceeds 8 MB upload limit")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("must be UTF-8") from exc
    if suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        # Blank optional cells represent absent values, not missing mandatory fields.
        optional = {"reference_answer", "evidence", "difficulty", "generator_model"}
        rows = [{k: (None if k in optional and v == "" else v) for k, v in row.items()}
                for row in rows]
    elif suffix.lower() == ".jsonl":
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {exc}") from exc
    else:
        raise ValueError("only .csv or .jsonl supported")
    if len(rows) > 20000:
        raise ValueError("at most 20,000 items per import")
    items, ids, digests = [], set(), set()
    for n, row in enumerate(rows, 1):
        try:
            item = EvaluationItem.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"record {n}: {exc}") from exc
        digest = _digest(item)
        if item.evaluation_id in ids or digest in digests:
            raise ValueError(f"record {n}: duplicate ID or identical question-answer pair")
        ids.add(item.evaluation_id)
        digests.add(digest)
        items.append(item)
    return items


def load_path(path: str | Path) -> list[EvaluationItem]:
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("file exceeds 8 MB upload limit")
    return load_bytes(path.read_bytes(), path.suffix)


def export_items(items: list[EvaluationItem], fmt: str) -> bytes:
    rows = [i.model_dump(mode="json") for i in items]
    if fmt == "jsonl":
        return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
    if fmt == "csv":
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=list(EvaluationItem.model_fields))
        writer.writeheader()
        writer.writerows(rows)
        return f.getvalue().encode()
    raise ValueError("export format must be csv or jsonl")
