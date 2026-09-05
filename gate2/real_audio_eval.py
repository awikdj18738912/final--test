#!/usr/bin/env python3
"""Evaluate real-audio Gate2 revisions against a small manually curated manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .live_smoke import run as run_session, write_json


ROOT = Path(__file__).resolve().parent


def normalize(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?；;、：:‘’“”\"'（）()【】\[\]{}<>《》…—_-~`]+", "", text)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dataset_path = Path(manifest["dataset"])
    rows = {
        line_no: json.loads(line)
        for line_no, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1)
    }
    records = []
    for item in manifest["records"]:
        line_no = int(item["line"])
        if line_no not in rows:
            raise ValueError(f"manifest line does not exist: {line_no}")
        row = rows[line_no]
        audio = Path(row["audio_filepath"])
        if not audio.exists():
            raise FileNotFoundError(audio)
        records.append({
            "id": item["id"],
            "line": line_no,
            "category": item.get("category", "unclassified"),
            "annotation_status": item.get("annotation_status", "curated"),
            "expected_clean_source": item.get("expected_clean_source", "manual_or_reference"),
            "audio": audio,
            "duration": float(row["duration"]),
            "reference": str(row["text"]),
            "expected_clean": str(item.get("expected_clean", row["text"])),
        })
    return records


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = load_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    if args.ids:
        wanted = set(args.ids)
        records = [item for item in records if item["id"] in wanted]
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(records, 1):
        result = await run_session(SimpleNamespace(
            url=args.url,
            audio=item["audio"],
            audio_sec=args.audio_sec,
            frame_ms=args.frame_ms,
            timeout=args.timeout,
        ))
        raw = str(result.get("final_raw_text") or "")
        visible = str(result.get("final_visible_text") or "")
        expected = str(item["expected_clean"])
        raw_norm, visible_norm, expected_norm = map(normalize, (raw, visible, expected))
        raw_error = edit_distance(raw_norm, expected_norm)
        visible_error = edit_distance(visible_norm, expected_norm)
        refinements = result.get("session_snapshot", {}).get("refinement_results", [])
        reject_reasons = [
            reason
            for refinement in refinements
            if refinement.get("event") == "refiner_reject"
            for reason in refinement.get("validation", {}).get("reasons", [])
        ]
        samples.append({
            "id": item["id"],
            "line": item["line"],
            "category": item["category"],
            "annotation_status": item["annotation_status"],
            "expected_clean_source": item["expected_clean_source"],
            "audio": str(item["audio"]),
            "duration": item["duration"],
            "reference": item["reference"],
            "expected_clean": expected,
            "raw_text": raw,
            "visible_text": visible,
            "raw_error": raw_error,
            "visible_error": visible_error,
            "raw_cer": round(raw_error / max(1, len(expected_norm)), 6),
            "visible_cer": round(visible_error / max(1, len(expected_norm)), 6),
            "changed_from_raw": visible_norm != raw_norm,
            "negative_edit": visible_error > raw_error,
            "revision_events": int(result.get("revision_events", 0)),
            "reject_reasons": sorted(set(reject_reasons)),
            "status": result.get("status"),
        })
        print(f"[{index}/{len(records)}] {item['id']} raw={raw_error} visible={visible_error} "
              f"revision={samples[-1]['revision_events']} reject={samples[-1]['reject_reasons']}")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_category[sample["category"]].append(sample)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(values),
            "raw_cer": round(sum(item["raw_error"] for item in values) / max(1, sum(len(normalize(item["expected_clean"])) for item in values)), 6),
            "visible_cer": round(sum(item["visible_error"] for item in values) / max(1, sum(len(normalize(item["expected_clean"])) for item in values)), 6),
            "revision_rate": round(sum(item["revision_events"] > 0 for item in values) / max(1, len(values)), 6),
            "negative_edit_rate": round(sum(item["negative_edit"] for item in values) / max(1, len(values)), 6),
            "changed_rate": round(sum(item["changed_from_raw"] for item in values) / max(1, len(values)), 6),
            "reject_rate": round(sum(bool(item["reject_reasons"]) for item in values) / max(1, len(values)), 6),
        }

    return {
        "benchmark": "Gate2 real-audio quality evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if samples and not any(item["status"] != "pass" for item in samples) else "fail",
        "service": args.url,
        "manifest": str(args.manifest.resolve()),
        "sample_count": len(samples),
        "annotation_status_counts": dict(Counter(item["annotation_status"] for item in samples)),
        "reviewed_sample_count": sum(item["annotation_status"] == "curated" for item in samples),
        "pending_manual_review_count": sum(item["annotation_status"] != "curated" for item in samples),
        "quality_ready": all(item["annotation_status"] == "curated" for item in samples),
        "summary": summarize(samples),
        "by_category": {key: summarize(value) for key, value in sorted(by_category.items())},
        "reject_reasons": dict(Counter(reason for item in samples for reason in item["reject_reasons"])),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--manifest", type=Path, default=ROOT / "real_audio_manifest.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--audio-sec", type=float)
    parser.add_argument("--frame-ms", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "real_audio_eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(evaluate(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "real_audio_eval.json", result)
    write_json(args.output_dir / "real_audio_eval_latest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"samples"}}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
