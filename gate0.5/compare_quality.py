#!/usr/bin/env python3
"""Create paired Gate 0.5 quality comparisons from backend result JSON."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = {
    "qwen_1_7b": ROOT / "results" / "quality" / "qwen17" / "qwen-streaming_latest.json",
    "qwen_0_6b": ROOT / "results" / "quality" / "qwen06" / "qwen-streaming_latest.json",
    "paraformer_online": ROOT / "results" / "quality" / "para" / "paraformer-online_latest.json",
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_comparison(
    name_a: str,
    samples_a: list[dict[str, Any]],
    name_b: str,
    samples_b: list[dict[str, Any]],
    bootstrap_iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    by_id_a = {item["utterance_id"]: item for item in samples_a}
    by_id_b = {item["utterance_id"]: item for item in samples_b}
    ids = sorted(by_id_a)
    if ids != sorted(by_id_b):
        raise ValueError(f"sample IDs differ between {name_a} and {name_b}")
    rows = []
    for utterance_id in ids:
        a, b = by_id_a[utterance_id], by_id_b[utterance_id]
        if a["reference_chars"] != b["reference_chars"] or a["normalized_reference"] != b["normalized_reference"]:
            raise ValueError(f"reference differs for {utterance_id}")
        rows.append((int(a["edit_distance"]), int(b["edit_distance"]), int(a["reference_chars"])))
    ref_chars = sum(item[2] for item in rows)
    observed = (sum(item[0] for item in rows) - sum(item[1] for item in rows)) / ref_chars
    deltas = []
    for _ in range(bootstrap_iterations):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        draw_chars = sum(item[2] for item in draw)
        deltas.append((sum(item[0] for item in draw) - sum(item[1] for item in draw)) / draw_chars)
    return {
        "a": name_a,
        "b": name_b,
        "definition": "negative CER delta favors a; win means fewer character edits for a on that utterance",
        "utterances": len(rows),
        "a_wins": sum(a < b for a, b, _ in rows),
        "ties": sum(a == b for a, b, _ in rows),
        "b_wins": sum(a > b for a, b, _ in rows),
        "corpus_cer_delta_a_minus_b": round(observed, 6),
        "paired_bootstrap_95_ci": [round(percentile(deltas, 0.025), 6), round(percentile(deltas, 0.975), 6)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "quality" / "quality_comparison.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--result", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    paths = dict(DEFAULT_RESULTS)
    for value in args.result:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--result must be NAME=PATH")
        paths[name] = Path(path)
    if args.bootstrap_iterations <= 0:
        parser.error("bootstrap-iterations must be positive")

    results = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    hashes = {item["manifest_source_sha256"] for item in results.values()}
    manifests = {item["manifest"] for item in results.values()}
    if len(hashes) != 1 or len(manifests) != 1:
        raise ValueError("results do not use the same source manifest")
    for name, item in results.items():
        if item["status"] != "pass":
            raise ValueError(f"{name} result did not pass")

    summary = {}
    for name, item in results.items():
        summary[name] = {
            "result_path": str(paths[name].resolve()),
            "model_path": item["backend"]["model_path"],
            "samples": item["summary"]["samples_total"],
            "corpus_cer": item["summary"]["corpus_cer"],
            "macro_cer_mean": item["summary"]["macro_cer"]["mean"],
            "compute_rtf_mean": item["summary"]["compute_rtf"]["mean"],
            "compute_rtf_p95": item["summary"]["compute_rtf"]["p95"],
            "by_stratum_corpus_cer": {
                stratum: values["corpus_cer"] for stratum, values in item["summary"]["by_stratum"].items()
            },
        }

    names = list(results)
    rng = random.Random(args.seed)
    comparisons = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            comparisons.append(
                paired_comparison(
                    name_a,
                    results[name_a]["samples"],
                    name_b,
                    results[name_b]["samples"],
                    args.bootstrap_iterations,
                    rng,
                )
            )
    output = {
        "comparison": "Gate 0.5 paired stratified ASR quality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest": next(iter(manifests)),
        "manifest_source_sha256": next(iter(hashes)),
        "bootstrap": {"iterations": args.bootstrap_iterations, "seed": args.seed, "unit": "utterance"},
        "backends": summary,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
