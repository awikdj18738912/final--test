#!/usr/bin/env python3
"""Create a spreadsheet-friendly manual annotation sheet from a Gate2 manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "real_audio_manifest_expanded.json")
    parser.add_argument("--output", type=Path, default=ROOT / "real_audio_annotation_sheet.csv")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset"])
    rows = {
        line_no: json.loads(line)
        for line_no, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1)
    }
    fields = [
        "id", "line", "audio_path", "duration_sec", "current_category",
        "annotation_status", "reference_text", "expected_clean",
        "decision", "final_category", "reviewer_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in manifest["records"]:
            row = rows[int(item["line"])]
            writer.writerow({
                "id": item["id"],
                "line": item["line"],
                "audio_path": row["audio_filepath"],
                "duration_sec": row["duration"],
                "current_category": item.get("category", "unclassified"),
                "annotation_status": item.get("annotation_status", "curated"),
                "reference_text": row["text"],
                "expected_clean": item.get("expected_clean", row["text"]),
                "decision": "" if item.get("annotation_status") != "curated" else "keep",
                "final_category": item.get("category", "unclassified"),
                "reviewer_notes": "",
            })
    print(f"wrote {len(manifest['records'])} rows to {args.output}")
    print("Fill expected_clean, decision (keep/correct/reject), final_category, and reviewer_notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
