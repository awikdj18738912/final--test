#!/usr/bin/env python3
"""Build a deterministic, reviewable expansion of the Gate2 real-audio manifest.

The category rules are only sampling aids.  Every generated record is marked
``heuristic_pending_manual_review`` so it cannot be mistaken for a gold label.
Existing curated records are copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def has_adjacent_repeat(text: str) -> bool:
    return bool(re.search(r"(.)\1|(.{2,4})\2", text))


def category_candidates(text: str) -> dict[str, bool]:
    return {
        "self_correction": bool(re.search(r"不对|不是.{0,8}是|应该是|准确地?说|我的意思|重新说|换句话说", text)),
        "repetition": has_adjacent_repeat(text),
        "filler": bool(re.search(r"呃+|嗯+|啊+|那个|就是|然后然后|这个这个", text)),
        "numbers": bool(re.search(r"[0-9０-９一二三四五六七八九十百千万亿年月日%％]", text)),
        "english_mix": bool(re.search(r"[A-Za-z]", text)),
    }


def load_rows(dataset: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]


def choose(rows: list[dict[str, Any]], excluded: set[int], category: str, count: int) -> list[tuple[int, dict[str, Any]]]:
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for line_no, row in enumerate(rows, 1):
        if line_no in excluded:
            continue
        text = str(row.get("text", ""))
        tags = category_candidates(text)
        if category == "passthrough":
            matched = not any(tags.values())
        else:
            matched = tags[category]
        if not matched:
            continue
        # Stable ordering spreads duration and source utterances without using
        # random state.  The hash is only a reproducible tie-breaker.
        key = hashlib.sha256(f"{category}:{line_no}".encode()).hexdigest()
        candidates.append((key, line_no, row))
    candidates.sort(key=lambda item: item[0])
    return [(line_no, row) for _, line_no, row in candidates[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/aim0/data/datasets/ASR/wenetspeech_test_net/wenetspeech_test_net.json"))
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "real_audio_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "real_audio_manifest_expanded.json")
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    rows = load_rows(args.source)
    records = list(base["records"])
    excluded = {int(item["line"]) for item in records}
    target_categories = ["passthrough", "repetition", "filler", "self_correction", "numbers", "english_mix"]
    counts: dict[str, int] = {}
    next_number = 1
    for category in target_categories:
        selected = choose(rows, excluded, category, args.per_category)
        counts[category] = len(selected)
        for line_no, row in selected:
            records.append({
                "id": f"heuristic_{category}_{next_number:04d}",
                "line": line_no,
                "category": category,
                "annotation_status": "heuristic_pending_manual_review",
                "expected_clean_source": "reference_default",
            })
            next_number += 1
            excluded.add(line_no)

    output = {
        "dataset": str(args.source),
        "dataset_note": (
            "Generated deterministically from the local WenetSpeech test-net cache. "
            "Heuristic categories are sampling aids only; records marked "
            "heuristic_pending_manual_review must be manually reviewed before quality claims."
        ),
        "base_manifest": str(args.base_manifest.resolve()),
        "selection": {"per_category_requested": args.per_category, "added_by_category": counts},
        "records": records,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total": len(records), "added": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
