#!/usr/bin/env python3
"""Replay a completed Gate2 evaluation through the conservative Gate3 router.

This is an offline counterfactual: a routed sample reuses the already recorded
Refiner output when the router would have called it, otherwise it keeps raw
ASR text.  It neither starts the service nor uses a GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gate2.real_audio_eval import edit_distance, normalize
from gate2.live_smoke import write_json
from gate3.rule_router import route


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / "gate2" / "results" / "real_audio_eval_curated" / "real_audio_eval_latest.json"


def _error(text: str, expected: str) -> int:
    return edit_distance(normalize(text), normalize(expected))


def _summarize(samples: list[dict[str, Any]], output_key: str, call_mode: str) -> dict[str, Any]:
    expected_chars = sum(max(1, len(normalize(item["expected_clean"]))) for item in samples)
    error = sum(item[output_key + "_error"] for item in samples)
    raw_error = sum(item["raw_error"] for item in samples)
    if call_mode == "none":
        calls = 0
    elif call_mode == "all":
        calls = len(samples)
    elif call_mode == "route":
        calls = sum(item["route"]["call_refiner"] for item in samples)
    else:
        raise ValueError(f"unknown call mode: {call_mode}")
    return {
        "samples": len(samples),
        "cer": round(error / max(1, expected_chars), 6),
        "error_characters": error,
        "raw_cer": round(raw_error / max(1, expected_chars), 6),
        "delta_cer_vs_raw": round((error - raw_error) / max(1, expected_chars), 6),
        "improved_rate": round(sum(item[output_key + "_error"] < item["raw_error"] for item in samples) / max(1, len(samples)), 6),
        "negative_edit_rate": round(sum(item[output_key + "_error"] > item["raw_error"] for item in samples) / max(1, len(samples)), 6),
        "changed_from_raw_rate": round(sum(item[output_key + "_text"] != item["raw_text"] for item in samples) / max(1, len(samples)), 6),
        "refiner_call_rate": round(calls / max(1, len(samples)), 6),
    }


def evaluate(input_path: Path) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for original in source["samples"]:
        raw = str(original.get("raw_text") or "")
        visible = str(original.get("visible_text") or "")
        expected = str(original["expected_clean"])
        decision = route(raw)
        routed = visible if decision.call_refiner else raw
        item = {
            "id": original["id"],
            "category": original.get("category", "unclassified"),
            "decision": original.get("decision", ""),
            "expected_clean": expected,
            "raw_text": raw,
            "refine_all_text": visible,
            "rule_route_text": routed,
            "raw_error": _error(raw, expected),
            "refine_all_error": _error(visible, expected),
            "rule_route_error": _error(routed, expected),
            "route": decision.to_dict(),
        }
        samples.append(item)

    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        by_decision[item["decision"] or "unspecified"].append(item)
    reasons = Counter(reason.split(":", 1)[0] for item in samples for reason in item["route"]["reasons"])

    return {
        "benchmark": "Gate3 conservative rule-router replay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_evaluation": str(input_path.resolve()),
        "method": "offline counterfactual; selected samples reuse recorded Gate2 visible_text",
        "sample_count": len(samples),
        "comparisons": {
            "skip_all": _summarize(samples, "raw", "none"),
            "refine_all": _summarize(samples, "refine_all", "all"),
            "rule_route": _summarize(samples, "rule_route", "route"),
        },
        "by_manual_decision": {
            key: {
                "skip_all": _summarize(value, "raw", "none"),
                "refine_all": _summarize(value, "refine_all", "all"),
                "rule_route": _summarize(value, "rule_route", "route"),
                "rule_call_coverage": round(sum(item["route"]["call_refiner"] for item in value) / max(1, len(value)), 6),
            }
            for key, value in sorted(by_decision.items())
        },
        "reason_distribution": dict(sorted(reasons.items())),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rule_route")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args.input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "rule_route_eval.json", result)
    write_json(args.output_dir / "rule_route_latest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
