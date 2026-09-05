#!/usr/bin/env python3
"""Evaluate raw and safety-committed Refiner text on AASR-Bench oral/clean pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import PatchCompiler, PatchValidator, SourceWindow, ValidationResult, text_hash
from .refiner import TransformersRefiner


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/home/aim0/data/datasets/ASR/AASR-Bench/data/benchmark.jsonl")
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ref_chars = sum(item["reference_chars"] for item in records)
    source_errors = sum(item["source_edit_distance"] for item in records)
    model_errors = sum(item["model_edit_distance"] for item in records)
    committed_errors = sum(item["committed_edit_distance"] for item in records)
    model_changes = [item for item in records if item["normalized_model"] != item["normalized_oral"]]
    committed_changes = [item for item in records if item["decision"] == "replace"]
    rejected = [item for item in records if item["decision"] == "reject"]
    reason_counts = Counter(reason for item in rejected for reason in item["validation"]["reasons"])
    return {
        "samples": len(records),
        "reference_chars": ref_chars,
        "source_corpus_cer": round(source_errors / ref_chars, 6) if ref_chars else None,
        "model_corpus_cer": round(model_errors / ref_chars, 6) if ref_chars else None,
        "committed_corpus_cer": round(committed_errors / ref_chars, 6) if ref_chars else None,
        "model_improved": sum(item["model_edit_distance"] < item["source_edit_distance"] for item in records),
        "model_regressed": sum(item["model_edit_distance"] > item["source_edit_distance"] for item in records),
        "committed_improved": sum(item["committed_edit_distance"] < item["source_edit_distance"] for item in records),
        "committed_regressed": sum(item["committed_edit_distance"] > item["source_edit_distance"] for item in records),
        "accepted": sum(item["validation"]["accepted"] for item in records),
        "keep": sum(item["decision"] == "keep" for item in records),
        "rejected": sum(item["decision"] == "reject" for item in records),
        "model_substantive_change": sum(item["normalized_model"] != item["normalized_oral"] for item in records),
        "committed_substantive_change": sum(item["normalized_committed"] != item["normalized_oral"] for item in records),
        "model_negative_correction_rate": round(
            sum(item["model_edit_distance"] > item["source_edit_distance"] for item in model_changes)
            / len(model_changes),
            6,
        ) if model_changes else None,
        "committed_negative_correction_rate": round(
            sum(item["committed_edit_distance"] > item["source_edit_distance"] for item in committed_changes)
            / len(committed_changes),
            6,
        ) if committed_changes else None,
        "accepted_change_improved": sum(
            item["model_edit_distance"] < item["source_edit_distance"] for item in committed_changes
        ),
        "accepted_change_equal": sum(
            item["model_edit_distance"] == item["source_edit_distance"] for item in committed_changes
        ),
        "accepted_change_regressed": sum(
            item["model_edit_distance"] > item["source_edit_distance"] for item in committed_changes
        ),
        "rejected_improved": sum(item["model_edit_distance"] < item["source_edit_distance"] for item in rejected),
        "rejected_equal": sum(item["model_edit_distance"] == item["source_edit_distance"] for item in rejected),
        "rejected_regressed": sum(item["model_edit_distance"] > item["source_edit_distance"] for item in rejected),
        "validation_reason_counts": dict(sorted(reason_counts.items())),
        "mean_amortized_inference_sec": round(statistics.fmean(item["amortized_inference_sec"] for item in records), 6) if records else None,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "aasr_bench")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_every <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch-size, checkpoint-every and limit must be positive")

    source_bytes = args.data.read_bytes()
    records = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    records = [item for item in records if item.get("language") == args.language]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("no benchmark records selected")

    refiner = TransformersRefiner(args.model, max_new_tokens=args.max_new_tokens)
    refiner.rewrite(read_only_prefix="", active_source_window=records[0]["oral"], trusted_memory=[])
    compiler = PatchCompiler()
    validator = PatchValidator(max_patches=32, max_changed_chars=240, max_change_ratio=0.75)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    result: dict[str, Any] = {
        "benchmark": "AASR-Bench deterministic text metrics with safety commit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": "oral-to-clean Refiner evaluation; no audio ASR and no LLM rubric judge",
        "license_note": "dataset card asserts no license; local research cache only, do not redistribute",
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "language": args.language,
        "model": str(args.model.resolve()),
        "model_role": "offline/heavy baseline only",
        "load_sec": refiner.load_sec,
        "batch_size": args.batch_size,
        "requested_samples": len(records),
        "samples": [],
    }
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        generated = refiner.rewrite_batch(
            [{"read_only_prefix": "", "active_source_window": item["oral"], "trusted_memory": []} for item in batch]
        )
        for item, refined in zip(batch, generated):
            oral, clean, model_text = item["oral"], item["clean"], refined["text"]
            window_id = "aasr_" + hashlib.sha256(item["source_record_id"].encode("utf-8")).hexdigest()[:16]
            window = SourceWindow(window_id, (item["source_record_id"],), oral, oral, "", 0, text_hash(oral))
            patch_set = compiler.diff(window=window, clean_text=model_text, correction_type="offline_refine")
            pass_through = item["scene"] == "passthrough"
            validation = validator.validate(
                patch_set=patch_set,
                window=window,
                clean_text=model_text,
                tenant_id="aasr-bench",
                evidence=[],
                pass_through=pass_through,
            )
            unsafe_truncation = (
                refined["finish_reason"] == "length"
                and not refined["key_suffix_present"]
                and model_text != oral
            )
            if unsafe_truncation:
                validation = ValidationResult(False, tuple(dict.fromkeys((*validation.reasons, "generation_max_tokens"))))
            decision = "keep" if validation.accepted and not patch_set.patches else "replace" if validation.accepted else "reject"
            committed = model_text if validation.accepted else oral
            normalized_clean = normalize(clean)
            normalized_oral = normalize(oral)
            normalized_model = normalize(model_text)
            normalized_committed = normalize(committed)
            sample = {
                "source_record_id": item["source_record_id"],
                "scene": item["scene"],
                "language": item["language"],
                "oral": oral,
                "clean": clean,
                "raw_model_text": refined["raw_text"],
                "model_text": model_text,
                "model_keys": refined["keys"],
                "model_finish_reason": refined["finish_reason"],
                "model_key_suffix_present": refined["key_suffix_present"],
                "model_incomplete_key_suffix": refined["incomplete_key_suffix"],
                "decision": decision,
                "validation": asdict(validation),
                "patches": [asdict(value) for value in patch_set.patches],
                "committed_text": committed,
                "normalized_oral": normalized_oral,
                "normalized_model": normalized_model,
                "normalized_committed": normalized_committed,
                "reference_chars": len(normalized_clean),
                "source_edit_distance": edit_distance(normalized_clean, normalized_oral),
                "model_edit_distance": edit_distance(normalized_clean, normalized_model),
                "committed_edit_distance": edit_distance(normalized_clean, normalized_committed),
                "batch_inference_sec": refined["batch_inference_sec"],
                "amortized_inference_sec": refined["amortized_inference_sec"],
            }
            result["samples"].append(sample)
        completed = len(result["samples"])
        if completed % args.checkpoint_every < args.batch_size or completed == len(records):
            result["progress"] = {"completed": completed, "total": len(records)}
            write_json(run_dir / "aasr_bench.partial.json", result)

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in result["samples"]:
        by_scene[item["scene"]].append(item)
    result["summary"] = summarize(result["samples"])
    result["summary"]["by_scene"] = {name: summarize(items) for name, items in sorted(by_scene.items())}
    pass_through_records = by_scene.get("passthrough", [])
    result["summary"]["pass_through"] = {
        **summarize(pass_through_records),
        "model_over_edit_rate": round(sum(item["normalized_model"] != item["normalized_oral"] for item in pass_through_records) / len(pass_through_records), 6) if pass_through_records else None,
        "committed_over_edit_rate": round(sum(item["normalized_committed"] != item["normalized_oral"] for item in pass_through_records) / len(pass_through_records), 6) if pass_through_records else None,
    }
    result["status"] = "pass" if len(result["samples"]) == len(records) else "fail"
    result["progress"] = {"completed": len(records), "total": len(records)}
    write_json(run_dir / "aasr_bench.json", result)
    write_json(args.output_dir / "aasr_bench_latest.json", result)
    print(json.dumps({"status": result["status"], "model": result["model"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
