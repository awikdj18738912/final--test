#!/usr/bin/env python3
"""Compare independent holdout runs from conservative routing and all-refine."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gate2.live_smoke import write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_ROUTE = ROOT / "results" / "holdout_route" / "real_audio_eval_latest.json"
DEFAULT_ALL = ROOT / "results" / "holdout_all" / "real_audio_eval_latest.json"


def _router_counts(samples: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if all("router_call_count" in item for item in samples):
        return {
            "router_mode": mode,
            "refiner_calls": sum(item["router_call_count"] for item in samples),
            "router_skips": sum(item["router_skip_count"] for item in samples),
            "refiner_rpcs": sum(item["refiner_rpc_count"] for item in samples),
        }
    if mode == "all":
        return {
            "router_mode": "all",
            "refiner_calls": None,
            "router_skips": None,
            "refiner_rpcs": None,
            "note": "This pre-instrumentation artifact does not retain counts; all closed windows were selected by configuration.",
        }
    return {
        "router_mode": mode,
        "refiner_calls": None,
        "router_skips": None,
        "refiner_rpcs": None,
        "note": "This pre-instrumentation artifact does not retain router counts.",
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reference_chars = sum(max(1, len("".join(ch for ch in item["expected_clean"] if ch.isalnum()))) for item in samples)
    raw_error = sum(item["raw_error"] for item in samples)
    visible_error = sum(item["visible_error"] for item in samples)
    return {
        "samples": len(samples),
        "raw_cer": round(raw_error / max(1, reference_chars), 6),
        "output_cer": round(visible_error / max(1, reference_chars), 6),
        "delta_cer_vs_raw": round((visible_error - raw_error) / max(1, reference_chars), 6),
        "negative_edit_rate": round(sum(item["negative_edit"] for item in samples) / max(1, len(samples)), 6),
        "changed_rate": round(sum(item["changed_from_raw"] for item in samples) / max(1, len(samples)), 6),
    }


def _by_decision(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample.get("decision") or "unspecified"].append(sample)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def compare(route_path: Path, all_path: Path) -> dict[str, Any]:
    routed = json.loads(route_path.read_text(encoding="utf-8"))
    all_refined = json.loads(all_path.read_text(encoding="utf-8"))
    route_by_id = {item["id"]: item for item in routed["samples"]}
    all_by_id = {item["id"]: item for item in all_refined["samples"]}
    if set(route_by_id) != set(all_by_id):
        raise ValueError("route and all-refine runs do not contain the same sample IDs")
    raw_mismatches = [
        record_id for record_id in route_by_id
        if route_by_id[record_id]["raw_text"] != all_by_id[record_id]["raw_text"]
    ]
    return {
        "benchmark": "Gate3 independent-holdout policy comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route_run": str(route_path.resolve()),
        "all_refine_run": str(all_path.resolve()),
        "sample_count": len(route_by_id),
        "raw_transcript_mismatch_count": len(raw_mismatches),
        "raw_transcript_mismatch_ids": raw_mismatches,
        "policies": {
            "conservative_route": {
                "router": _router_counts(routed["samples"], "conservative"),
                "overall": _summary(routed["samples"]),
                "by_manual_decision": _by_decision(routed["samples"]),
            },
            "refine_all": {
                "router": _router_counts(all_refined["samples"], "all"),
                "overall": _summary(all_refined["samples"]),
                "by_manual_decision": _by_decision(all_refined["samples"]),
            },
        },
        "interpretation": (
            "This is a held-out policy comparison. The conservative rule set must not be tuned "
            "against these results; use them to motivate a separately trained/validated benefit predictor."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--all-refine", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "holdout_comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(args.route, args.all_refine)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "holdout_comparison.json", result)
    write_json(args.output_dir / "holdout_comparison_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
