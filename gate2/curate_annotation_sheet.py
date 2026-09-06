#!/usr/bin/env python3
"""Validate a reviewed CSV and generate the immutable Gate2 evaluation manifest.

The source CSV remains the editable audit record.  This command normalizes the
two previously agreed label rules: ``keep`` implies ``passthrough``; and a
word-level change from reference to expected text implies ``correct``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SHEET_COLUMNS = [
    "id", "line", "audio_path", "duration_sec", "current_category",
    "annotation_status", "reference_text", "expected_clean",
    "decision", "final_category", "reviewer_notes",
]
DECISIONS = {"keep", "correct", "reject"}


def normalize(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?；;、：:‘’“”\"'（）()【】\[\]{}<>《》…—_-~`]+", "", text)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in SHEET_COLUMNS} for row in rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def validate_and_normalize(rows: list[dict[str, str]]) -> Counter[str]:
    ids: set[str] = set()
    changes: Counter[str] = Counter()
    errors: list[str] = []
    for row in rows:
        record_id = row.get("id", "")
        if not record_id or record_id in ids:
            errors.append(f"duplicate or blank id: {record_id!r}")
        ids.add(record_id)
        decision = row.get("decision", "").strip().lower()
        expected = row.get("expected_clean", "").strip()
        if decision not in DECISIONS:
            errors.append(f"{record_id}: unsupported decision {decision!r}")
            continue
        if decision in {"keep", "correct"} and not expected:
            errors.append(f"{record_id}: expected_clean is required for {decision}")
        if decision == "keep" and normalize(expected) != normalize(row.get("reference_text", "")):
            row["decision"] = "correct"
            changes["keep_to_correct_text_changed"] += 1
            decision = "correct"
        if decision == "keep" and row.get("final_category") != "passthrough":
            row["final_category"] = "passthrough"
            changes["keep_category_to_passthrough"] += 1
        row["annotation_status"] = "excluded" if decision == "reject" else "curated"
    if errors:
        raise ValueError("Annotation sheet is invalid:\n- " + "\n- ".join(errors))
    return changes


def build_manifest(rows: list[dict[str, str]], base_manifest: Path) -> dict[str, Any]:
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    source_by_id = {item["id"]: item for item in base["records"]}
    missing = [row["id"] for row in rows if row["id"] not in source_by_id]
    if missing:
        raise ValueError(f"annotation ids absent from base manifest: {missing[:5]}")
    records: list[dict[str, Any]] = []
    for row in rows:
        item = dict(source_by_id[row["id"]])
        item.update({
            "category": row["final_category"],
            "annotation_status": row["annotation_status"],
            "expected_clean": row["expected_clean"],
            "expected_clean_source": "manual",
            "decision": row["decision"],
            "evaluation_eligible": row["decision"] != "reject",
            "reviewer_notes": row["reviewer_notes"],
        })
        records.append(item)
    return {
        "dataset": base["dataset"],
        "dataset_note": "Human-reviewed Gate2 real-audio evaluation manifest.",
        "base_manifest": str(base_manifest.resolve()),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=ROOT / "real_audio_annotation_sheet.csv")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "real_audio_manifest_expanded.json")
    parser.add_argument("--output", type=Path, default=ROOT / "real_audio_manifest_curated.json")
    parser.add_argument("--apply", action="store_true", help="write normalized decisions and categories back to the source CSV")
    args = parser.parse_args()

    rows = read_sheet(args.sheet)
    changes = validate_and_normalize(rows)
    manifest = build_manifest(rows, args.base_manifest)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.apply:
        write_sheet(args.sheet, rows)
    summary = {
        "records": len(rows),
        "eligible": sum(item["evaluation_eligible"] for item in manifest["records"]),
        "excluded": sum(not item["evaluation_eligible"] for item in manifest["records"]),
        "decisions": dict(Counter(row["decision"] for row in rows)),
        "categories": dict(Counter(row["final_category"] for row in rows)),
        "normalizations": dict(changes),
        "output": str(args.output),
        "source_sheet_written": args.apply,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
