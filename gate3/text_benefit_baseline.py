#!/usr/bin/env python3
"""Development-only cross-validation of a text-only Refiner benefit predictor.

This experiment intentionally uses only signals that are available before a
Refiner call.  Its purpose is to reject an insufficient feature set, rather
than to deploy a classifier selected on the fixed Gate3 holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gate2.live_smoke import write_json
from gate3.rule_router import route


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / "gate2" / "results" / "real_audio_eval_curated" / "real_audio_eval_latest.json"
FEATURE_NAMES = (
    "bias",
    "text_length_scaled",
    "length_ge_30",
    "length_ge_60",
    "self_correction_phrase",
    "adjacent_repeat",
    "triple_repeat",
    "repeated_bigram",
    "filler_marker",
    "number_marker",
    "latin_marker",
    "punctuation_density",
    "rule_score_scaled",
)


def feature_vector(text: str) -> list[float]:
    """Return bounded, pre-Refiner text features without label information."""

    length = len(text)
    punctuation = len(re.findall(r"[，。！？,!.?]", text))
    rule_score = route(text).score
    return [
        1.0,
        min(length, 120) / 120,
        float(length >= 30),
        float(length >= 60),
        float(bool(re.search(r"不对|说错|不是.{0,24}[，,]是|应该是|我的意思|重新说", text))),
        float(bool(re.search(r"(.)\1", text))),
        float(bool(re.search(r"(.)\1{2,}", text))),
        float(bool(re.search(r"(.{2})\1", text))),
        float(bool(re.search(r"呃|嗯|啊|那个|就是", text))),
        float(bool(re.search(r"[0-9一二三四五六七八九十百千万亿]", text))),
        float(bool(re.search(r"[A-Za-z]", text))),
        min(punctuation, 10) / 10,
        min(rule_score, 4) / 4,
    ]


def stratified_folds(samples: list[dict[str, Any]], folds: int) -> list[int]:
    """Assign deterministic folds with the positive outcome spread across folds."""

    assignments = [0] * len(samples)
    for label in (False, True):
        indices = [
            index
            for index, item in enumerate(samples)
            if (item["visible_error"] < item["raw_error"]) == label
        ]
        indices.sort(key=lambda index: hashlib.sha256(samples[index]["id"].encode()).hexdigest())
        for offset, index in enumerate(indices):
            assignments[index] = offset % folds
    return assignments


def fit_logistic(features: list[list[float]], labels: list[int], *, epochs: int = 3000) -> list[float]:
    """Fit a small L2-regularized logistic model using only the standard library."""

    weights = [0.0] * len(features[0])
    learning_rate, l2 = 0.15, 0.1
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for vector, label in zip(features, labels):
            score = max(-30.0, min(30.0, sum(left * right for left, right in zip(weights, vector))))
            probability = 1.0 / (1.0 + math.exp(-score))
            error = probability - label
            for feature_index, value in enumerate(vector):
                gradient[feature_index] += error * value
        for feature_index in range(len(weights)):
            penalty = 0.0 if feature_index == 0 else l2 * weights[feature_index]
            weights[feature_index] -= learning_rate * (gradient[feature_index] / len(features) + penalty)
    return weights


def probability(weights: list[float], vector: list[float]) -> float:
    score = max(-30.0, min(30.0, sum(left * right for left, right in zip(weights, vector))))
    return 1.0 / (1.0 + math.exp(-score))


def evaluate(samples: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    vectors = [feature_vector(str(item["raw_text"])) for item in samples]
    labels = [int(item["visible_error"] < item["raw_error"]) for item in samples]
    assignments = stratified_folds(samples, folds)
    oof = [0.0] * len(samples)
    for fold in range(folds):
        train = [index for index, assigned in enumerate(assignments) if assigned != fold]
        test = [index for index, assigned in enumerate(assignments) if assigned == fold]
        weights = fit_logistic([vectors[index] for index in train], [labels[index] for index in train])
        for index in test:
            oof[index] = probability(weights, vectors[index])

    threshold_rows = []
    for threshold in [round(0.10 + step * 0.05, 2) for step in range(9)]:
        selected = [index for index, value in enumerate(oof) if value >= threshold]
        positives = sum(labels[index] for index in selected)
        net_error_reduction = sum(samples[index]["raw_error"] - samples[index]["visible_error"] for index in selected)
        threshold_rows.append({
            "threshold": threshold,
            "calls": len(selected),
            "call_rate": round(len(selected) / len(samples), 6),
            "true_improvements": positives,
            "precision": round(positives / len(selected), 6) if selected else None,
            "net_error_reduction": net_error_reduction,
        })

    viable = [
        row for row in threshold_rows
        if row["calls"] and row["precision"] is not None and row["precision"] >= 0.5 and row["net_error_reduction"] > 0
    ]
    return {
        "benchmark": "Gate3 development-only text benefit-prediction baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "folds": folds,
        "feature_names": list(FEATURE_NAMES),
        "target": "recorded full-Refiner visible_error < raw_error",
        "target_counts": dict(Counter("improved" if value else "not_improved" for value in labels)),
        "oof_probability_range": [round(min(oof), 6), round(max(oof), 6)],
        "thresholds": threshold_rows,
        "deployment_gate": {
            "minimum_precision": 0.5,
            "minimum_net_error_reduction": 1,
            "viable_thresholds": viable,
            "status": "pass" if viable else "reject",
            "reason": (
                "No threshold with at least one call achieves both the predeclared development "
                "precision and positive net-error-reduction gates. Do not deploy text-only routing."
                if not viable else "Development gate met; final decision still requires the fixed holdout."
            ),
        },
        "samples": [
            {
                "id": item["id"],
                "decision": item["decision"],
                "raw_text": item["raw_text"],
                "raw_error": item["raw_error"],
                "visible_error": item["visible_error"],
                "improved": bool(labels[index]),
                "oof_benefit_probability": round(oof[index], 6),
            }
            for index, item in enumerate(samples)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "text_benefit_baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    result = evaluate(source["samples"], args.folds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "text_benefit_baseline.json", result)
    write_json(args.output_dir / "text_benefit_baseline_latest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False, indent=2))
    return 0 if result["deployment_gate"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
