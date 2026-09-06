#!/usr/bin/env python3
"""Build a deterministic 50-audio Gate3 holdout set without label leakage.

The prior 155 reviewed records are excluded by WenetSpeech line number.  The
categories below are sampling strata only; every record remains pending until a
human reviewer listens and supplies its decision and clean text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gate2.build_real_audio_manifest import category_candidates, load_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/home/aim0/data/datasets/ASR/wenetspeech_test_net/wenetspeech_test_net.json")
DEFAULT_EXCLUDE = ROOT.parent / "gate2" / "real_audio_manifest_curated.json"

# 35 clean controls plus 15 deliberately varied potential-error cases.
STRATA: tuple[tuple[str, int], ...] = (
    ("passthrough", 35),
    ("repetition", 4),
    ("filler", 3),
    ("self_correction", 3),
    ("numbers", 3),
    ("english_mix", 2),
)


def select(
    rows: list[dict[str, Any]], excluded_lines: set[int], category: str, count: int
) -> list[tuple[int, dict[str, Any]]]:
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for line, row in enumerate(rows, 1):
        if line in excluded_lines:
            continue
        tags = category_candidates(str(row.get("text", "")))
        if (not any(tags.values())) if category == "passthrough" else (not tags[category]):
            continue
        audio_path = Path(str(row.get("audio_filepath", "")))
        if not audio_path.is_file():
            continue
        key = hashlib.sha256(f"gate3-holdout:{category}:{line}".encode()).hexdigest()
        candidates.append((key, line, row))
    candidates.sort(key=lambda item: item[0])
    return [(line, row) for _, line, row in candidates[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exclude-manifest", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--output", type=Path, default=ROOT / "real_audio_manifest_holdout.json")
    args = parser.parse_args()

    excluded_manifest = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
    excluded_lines = {int(item["line"]) for item in excluded_manifest["records"]}
    rows = load_rows(args.source)
    records: list[dict[str, Any]] = []
    actual_counts: dict[str, int] = {}
    for category, target in STRATA:
        chosen = select(rows, excluded_lines, category, target)
        if len(chosen) != target:
            raise ValueError(f"only found {len(chosen)}/{target} eligible {category} records")
        actual_counts[category] = len(chosen)
        for line, _ in chosen:
            record_number = len(records) + 1
            records.append({
                "id": f"gate3_holdout_{record_number:03d}",
                "line": line,
                "category": category,
                "annotation_status": "pending_manual_review",
                "expected_clean_source": "reference_default",
            })
            excluded_lines.add(line)

    output = {
        "dataset": str(args.source),
        "dataset_note": (
            "Independent Gate3 holdout. Selection strata only support coverage; "
            "manual review is required before any quality claim."
        ),
        "excluded_manifest": str(args.exclude_manifest.resolve()),
        "excluded_record_count": len(excluded_manifest["records"]),
        "selection": {"requested": dict(STRATA), "selected": actual_counts},
        "records": records,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "selection": actual_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
