#!/usr/bin/env python3
"""Run the local heavy Refiner through the deterministic Gate 2 pipeline."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import BoundaryManager, Evidence, PatchCompiler, PatchValidator, VersionedWindowStore
from .refiner import TransformersRefiner


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")


def normalize(value: str) -> str:
    return re.sub(r"[\W_]", "", value, flags=re.UNICODE).lower()


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref_char in enumerate(reference, start=1):
        current = [index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    return edit_distance(reference, hypothesis) / len(reference) if reference else float(bool(hypothesis))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scenarios", type=Path, default=ROOT / "scenarios.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    suite = json.loads(args.scenarios.read_text(encoding="utf-8"))
    refiner = TransformersRefiner(args.model, max_new_tokens=args.max_new_tokens)
    compiler = PatchCompiler()
    validator = PatchValidator()
    records = []
    for scenario in suite["scenarios"]:
        manager = BoundaryManager(max_chars=80)
        for source in scenario["source_spans"]:
            manager.feed(source, is_endpoint=True)
        window = manager.select_window(k=int(scenario.get("k", 3)), max_chars=240)
        evidence = [
            Evidence(
                item["evidence_id"],
                suite["tenant_id"],
                item["text"],
                status=item.get("status", "confirmed"),
                trust=float(item.get("trust", 1.0)),
            )
            for item in scenario.get("trusted_memory", [])
        ]
        refined = refiner.rewrite(
            read_only_prefix=window.read_only_prefix,
            active_source_window=window.source_text,
            trusted_memory=[item.text for item in evidence],
        )
        patch_set = compiler.diff(
            window=window,
            clean_text=refined["text"],
            evidence_ids=[item.evidence_id for item in evidence],
            correction_type=scenario["correction_type"],
        )
        validation = validator.validate(
            patch_set=patch_set,
            window=window,
            clean_text=refined["text"],
            tenant_id=suite["tenant_id"],
            evidence=evidence,
            pass_through=bool(scenario.get("pass_through", False)),
        )
        store = VersionedWindowStore()
        snapshot = store.create(window.window_id, window.current_text, version=window.base_version)
        event = None
        final_text = window.current_text
        if validation.accepted:
            commit_patch = compiler.diff(
                window=snapshot,
                clean_text=refined["text"],
                evidence_ids=patch_set.evidence_ids,
                correction_type=patch_set.correction_type,
            )
            event = store.commit(commit_patch, refined["text"])
            final_text = event.text
        records.append(
            {
                "id": scenario["id"],
                "k": scenario.get("k", 3),
                "source_spans": scenario["source_spans"],
                "source_window": window.source_text,
                "read_only_prefix": window.read_only_prefix,
                "expected": scenario["expected"],
                "pass_through": bool(scenario.get("pass_through", False)),
                "trusted_memory": [asdict(item) for item in evidence],
                "raw_model_text": refined["raw_text"],
                "clean_window": refined["text"],
                "model_keys": refined["keys"],
                "inference_sec": refined["inference_sec"],
                "patches": [asdict(item) for item in patch_set.patches],
                "patch_hash": patch_set.patch_hash,
                "validation": asdict(validation),
                "revision_event": asdict(event) if event else None,
                "final_text": final_text,
                "source_cer": cer(scenario["expected"], window.source_text),
                "model_cer": cer(scenario["expected"], refined["text"]),
                "committed_cer": cer(scenario["expected"], final_text),
                "exact_match": normalize(final_text) == normalize(scenario["expected"]),
            }
        )

    passed = sum(item["exact_match"] for item in records)
    improved = sum(item["committed_cer"] < item["source_cer"] for item in records)
    regressed = sum(item["committed_cer"] > item["source_cer"] for item in records)
    safety_pass = regressed == 0 and all(item["exact_match"] for item in records if item["pass_through"])
    utility_pass = improved >= 2
    result = {
        "benchmark": "Gate 2 local heavy Refiner deterministic-pipeline smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if safety_pass and utility_pass else "fail",
        "scope": "six hand-authored smoke cases; not an AASR-Bench reproduction",
        "model": str(args.model.resolve()),
        "model_role": "offline/heavy baseline only",
        "load_sec": refiner.load_sec,
        "summary": {
            "cases": len(records),
            "exact_matches": passed,
            "improved": improved,
            "regressed": regressed,
            "accepted": sum(item["validation"]["accepted"] for item in records),
            "safety_pass": safety_pass,
            "utility_pass": utility_pass,
            "mean_inference_sec": statistics.fmean(item["inference_sec"] for item in records),
            "mean_source_cer": statistics.fmean(item["source_cer"] for item in records),
            "mean_committed_cer": statistics.fmean(item["committed_cer"] for item in records),
        },
        "cases": records,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "baseline.json", result)
    write_json(args.output_dir / "baseline_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
