#!/usr/bin/env python3
"""Local browser UI for manually reviewing Gate2 audio annotations.

Run on the remote host only.  The server binds to 127.0.0.1 by default and
serves audio through record IDs rather than exposing raw filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
SHEET_COLUMNS = [
    "id", "line", "audio_path", "duration_sec", "current_category",
    "annotation_status", "reference_text", "expected_clean",
    "decision", "final_category", "reviewer_notes",
]
DECISIONS = {"", "keep", "correct", "reject"}
CATEGORIES = {
    "passthrough", "repetition", "filler_repetition", "self_correction",
    "numbers", "english_mix", "other",
}


class AnnotationUpdate(BaseModel):
    expected_clean: str = ""
    decision: str = ""
    final_category: str = ""
    reviewer_notes: str = ""


def read_sheet(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in SHEET_COLUMNS} for row in rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def public_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        key: row.get(key, "")
        for key in SHEET_COLUMNS
        if key != "audio_path"
    }


def create_app(sheet: Path) -> FastAPI:
    app = FastAPI(title="Gate2 Audio Annotation")
    html_path = ROOT / "annotation_review.html"

    def records() -> list[dict[str, str]]:
        return read_sheet(sheet)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return html_path.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        rows = records()
        return {
            "status": "ok",
            "sheet": str(sheet),
            "records": len(rows),
            "curated": sum(row.get("annotation_status") == "curated" for row in rows),
        }

    @app.get("/api/records")
    def list_records() -> dict[str, Any]:
        return {"records": [public_record(row) for row in records()]}

    @app.get("/audio/{record_id}")
    def audio(record_id: str) -> FileResponse:
        row = next((item for item in records() if item.get("id") == record_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="record not found")
        path = Path(row["audio_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="audio file not found")
        return FileResponse(path, media_type="audio/wav", filename=f"{record_id}{path.suffix}")

    @app.post("/api/records/{record_id}")
    def update_record(record_id: str, update: AnnotationUpdate) -> dict[str, Any]:
        decision = update.decision.strip().lower()
        category = update.final_category.strip()
        if decision not in DECISIONS:
            raise HTTPException(status_code=422, detail="decision must be keep, correct, reject, or empty")
        if category and category not in CATEGORIES:
            raise HTTPException(status_code=422, detail=f"unsupported category: {category}")
        if decision in {"keep", "correct"} and not update.expected_clean.strip():
            raise HTTPException(status_code=422, detail="expected_clean is required for keep/correct")
        rows = records()
        row = next((item for item in rows if item.get("id") == record_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="record not found")
        row["expected_clean"] = update.expected_clean.strip()
        row["decision"] = decision
        row["final_category"] = category or row.get("current_category", "other")
        row["reviewer_notes"] = update.reviewer_notes.strip()
        if decision:
            row["annotation_status"] = "curated" if decision != "reject" else "excluded"
        write_sheet(sheet, rows)
        return {"record": public_record(row)}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=ROOT / "real_audio_annotation_sheet.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.sheet.resolve()), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
