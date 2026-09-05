#!/usr/bin/env python3
"""Approximate AgenticASR K-window streaming on AASR-Bench oral text.

This is a text-stream simulation: boundaries are inferred from the supplied oral
transcript. It does not reproduce VAD timing, evolving ASR hypotheses, or audio
latency and must not be reported as an end-to-end streaming benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aasr_bench_eval import edit_distance, normalize, write_json
from .core import BoundaryManager, PatchCompiler, PatchValidator, SourceWindow, text_hash
from .refiner import TransformersRefiner


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/home/aim0/data/datasets/ASR/AASR-Bench/data/benchmark.jsonl")
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")


def join_chunks(chunks: list[str]) -> str:
    """Join chunks using the spacing rule in AgenticASR's reference session."""
    output = ""
    for chunk in chunks:
        value = chunk.strip()
        if not value:
            continue
        if output and output[-1].isascii() and value[0].isascii():
            output += " "
        output += value
    return output


def chunk_oral(text: str, max_chars: int) -> list[str]:
    manager = BoundaryManager(max_chars=max_chars)
    return [span.text.strip() for span in manager.feed(text, is_final=True) if span.text.strip()]


def window_keys(chunk_count: int, ks: tuple[int, ...]) -> list[tuple[int, int]]:
    keys = {(index, index) for index in range(chunk_count)}
    keys.update((max(0, end - k + 1), end) for end in range(chunk_count) for k in ks)
    return sorted(keys)


def compose_update(
    chunks: list[str],
    outputs: dict[tuple[int, int], dict[str, Any]],
    *,
    end: int,
    k: int,
    field: str,
) -> str:
    start = max(0, end - k + 1)
    prefix = [outputs[(index, index)][field] for index in range(start)]
    return join_chunks([*prefix, outputs[(start, end)][field]])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_chars = sum(item["reference_chars"] for item in records)
    source_errors = sum(item["source_edit_distance"] for item in records)
    model_errors = sum(item["model_edit_distance"] for item in records)
    committed_errors = sum(item["committed_edit_distance"] for item in records)
    passthrough = [item for item in records if item["scene"] == "passthrough"]
    return {
        "samples": len(records),
        "reference_chars": reference_chars,
        "source_corpus_cer": round(source_errors / reference_chars, 6) if reference_chars else None,
        "model_corpus_cer": round(model_errors / reference_chars, 6) if reference_chars else None,
        "committed_corpus_cer": round(committed_errors / reference_chars, 6) if reference_chars else None,
        "model_improved": sum(item["model_edit_distance"] < item["source_edit_distance"] for item in records),
        "model_regressed": sum(item["model_edit_distance"] > item["source_edit_distance"] for item in records),
        "committed_improved": sum(item["committed_edit_distance"] < item["source_edit_distance"] for item in records),
        "committed_regressed": sum(item["committed_edit_distance"] > item["source_edit_distance"] for item in records),
        "passthrough_samples": len(passthrough),
        "model_passthrough_over_edit_rate": round(
            sum(item["normalized_model"] != item["normalized_oral"] for item in passthrough) / len(passthrough), 6
        ) if passthrough else None,
        "committed_passthrough_over_edit_rate": round(
            sum(item["normalized_committed"] != item["normalized_oral"] for item in passthrough) / len(passthrough), 6
        ) if passthrough else None,
        "mean_chunks": round(statistics.fmean(item["chunk_count"] for item in records), 6) if records else None,
        "mean_updates": round(statistics.fmean(item["update_count"] for item in records), 6) if records else None,
        "mean_estimated_model_calls": round(
            statistics.fmean(item["estimated_model_calls"] for item in records), 6
        ) if records else None,
        "mean_estimated_amortized_inference_sec": round(
            statistics.fmean(item["estimated_amortized_inference_sec"] for item in records), 6
        ) if records else None,
        "mean_model_instability": round(statistics.fmean(item["model_instability"] for item in records), 6)
        if records else None,
        "mean_committed_instability": round(
            statistics.fmean(item["committed_instability"] for item in records), 6
        ) if records else None,
        "window_accept_rate": round(
            sum(item["accepted_windows"] for item in records) / sum(item["unique_windows"] for item in records), 6
        ) if records and sum(item["unique_windows"] for item in records) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-chunk-chars", type=int, default=80)
    parser.add_argument("--k", type=int, action="append", dest="ks")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "streaming_text")
    args = parser.parse_args()
    ks = tuple(sorted(set(args.ks or [1, 2, 3])))
    if (
        args.batch_size <= 0
        or args.max_chunk_chars <= 0
        or args.checkpoint_every <= 0
        or any(k <= 0 for k in ks)
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("batch-size, max-chunk-chars, checkpoint-every, limit and K must be positive")

    source_bytes = args.data.read_bytes()
    records = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    records = [item for item in records if item.get("language") == args.language]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("no benchmark records selected")

    prepared: list[dict[str, Any]] = []
    work: list[tuple[int, tuple[int, int], str]] = []
    for sample_index, item in enumerate(records):
        chunks = chunk_oral(item["oral"], args.max_chunk_chars)
        if not chunks:
            raise ValueError(f"empty chunks for {item['source_record_id']}")
        prepared.append({"record": item, "chunks": chunks, "outputs": {}})
        for start, end in window_keys(len(chunks), ks):
            work.append((sample_index, (start, end), join_chunks(chunks[start : end + 1])))

    refiner = TransformersRefiner(args.model, max_new_tokens=args.max_new_tokens)
    refiner.rewrite(read_only_prefix="", active_source_window=work[0][2], trusted_memory=[])
    compiler = PatchCompiler()
    validator = PatchValidator(max_patches=32, max_changed_chars=240, max_change_ratio=0.75)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    result: dict[str, Any] = {
        "benchmark": "AASR-Bench deterministic text-stream K-window approximation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": (
            "oral-text boundary simulation only; no audio/VAD timing, evolving ASR hypotheses, "
            "LLM rubric judge, or end-to-end streaming latency"
        ),
        "reference_semantics": "AgenticASR StreamingRefinementSession window replacement",
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model": str(args.model.resolve()),
        "model_role": "offline/heavy baseline only",
        "language": args.language,
        "ks": list(ks),
        "max_chunk_chars": args.max_chunk_chars,
        "batch_size": args.batch_size,
        "load_sec": refiner.load_sec,
        "selected_samples": len(records),
        "unique_window_requests": len(work),
        "generated_windows": 0,
        "samples": [],
    }

    for offset in range(0, len(work), args.batch_size):
        batch = work[offset : offset + args.batch_size]
        generated = refiner.rewrite_batch(
            [
                {"read_only_prefix": "", "active_source_window": source, "trusted_memory": []}
                for _, _, source in batch
            ]
        )
        for (sample_index, key, source), refined in zip(batch, generated):
            item = prepared[sample_index]["record"]
            window_id = "stream_" + hashlib.sha256(
                f"{item['source_record_id']}:{key[0]}:{key[1]}".encode("utf-8")
            ).hexdigest()[:16]
            window = SourceWindow(window_id, (), source, source, "", 0, text_hash(source))
            patch_set = compiler.diff(window=window, clean_text=refined["text"], correction_type="streaming_refine")
            validation = validator.validate(
                patch_set=patch_set,
                window=window,
                clean_text=refined["text"],
                tenant_id="aasr-bench",
                evidence=[],
                pass_through=item["scene"] == "passthrough",
            )
            unsafe_truncation = (
                refined["finish_reason"] == "length"
                and not refined["key_suffix_present"]
                and refined["text"] != source
            )
            accepted = validation.accepted and not unsafe_truncation
            reasons = list(validation.reasons)
            if unsafe_truncation and "generation_max_tokens" not in reasons:
                reasons.append("generation_max_tokens")
            prepared[sample_index]["outputs"][key] = {
                "source": source,
                "model_text": refined["text"],
                "committed_text": refined["text"] if accepted else source,
                "accepted": accepted,
                "reasons": reasons,
                "finish_reason": refined["finish_reason"],
                "key_suffix_present": refined["key_suffix_present"],
                "incomplete_key_suffix": refined["incomplete_key_suffix"],
                "amortized_inference_sec": refined["amortized_inference_sec"],
            }
        result["generated_windows"] += len(batch)
        if result["generated_windows"] % args.checkpoint_every < args.batch_size:
            write_json(run_dir / "streaming_text.partial.json", result)

    for prepared_item in prepared:
        item = prepared_item["record"]
        chunks = prepared_item["chunks"]
        outputs = prepared_item["outputs"]
        normalized_clean = normalize(item["clean"])
        normalized_oral = normalize(item["oral"])
        for k in ks:
            model_revisions: list[str] = []
            committed_revisions: list[str] = []
            estimated_seconds = 0.0
            estimated_calls = 0
            for end in range(len(chunks)):
                start = max(0, end - k + 1)
                model_revisions.append(compose_update(chunks, outputs, end=end, k=k, field="model_text"))
                committed_revisions.append(
                    compose_update(chunks, outputs, end=end, k=k, field="committed_text")
                )
                estimated_seconds += outputs[(start, end)]["amortized_inference_sec"]
                estimated_calls += 1
                if end >= k:
                    estimated_seconds += outputs[(end - k, end - k)]["amortized_inference_sec"]
                    estimated_calls += 1
            model_text = model_revisions[-1]
            committed_text = committed_revisions[-1]
            normalized_model = normalize(model_text)
            normalized_committed = normalize(committed_text)

            def instability(revisions: list[str]) -> float:
                normalized = [normalize(value) for value in revisions]
                movement = sum(edit_distance(left, right) for left, right in zip(normalized, normalized[1:]))
                return round(movement / max(len(normalized[-1]), 1), 6)

            used_keys = {(max(0, end - k + 1), end) for end in range(len(chunks))}
            used_keys.update((index, index) for index in range(max(0, len(chunks) - k)))
            result["samples"].append(
                {
                    "source_record_id": item["source_record_id"],
                    "scene": item["scene"],
                    "k": k,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "update_count": len(chunks),
                    "unique_windows": len(used_keys),
                    "accepted_windows": sum(outputs[key]["accepted"] for key in used_keys),
                    "estimated_model_calls": estimated_calls,
                    "estimated_amortized_inference_sec": round(estimated_seconds, 6),
                    "oral": item["oral"],
                    "clean": item["clean"],
                    "model_text": model_text,
                    "committed_text": committed_text,
                    "normalized_oral": normalized_oral,
                    "normalized_model": normalized_model,
                    "normalized_committed": normalized_committed,
                    "reference_chars": len(normalized_clean),
                    "source_edit_distance": edit_distance(normalized_clean, normalized_oral),
                    "model_edit_distance": edit_distance(normalized_clean, normalized_model),
                    "committed_edit_distance": edit_distance(normalized_clean, normalized_committed),
                    "model_instability": instability(model_revisions),
                    "committed_instability": instability(committed_revisions),
                    "model_revisions": model_revisions,
                    "committed_revisions": committed_revisions,
                }
            )

    all_window_outputs = [
        output for prepared_item in prepared for output in prepared_item["outputs"].values()
    ]
    result["window_diagnostics"] = {
        "unique_windows": len(all_window_outputs),
        "accepted": sum(item["accepted"] for item in all_window_outputs),
        "finish_reason_counts": dict(sorted(Counter(item["finish_reason"] for item in all_window_outputs).items())),
        "key_suffix_present": sum(item["key_suffix_present"] for item in all_window_outputs),
        "incomplete_key_suffix": sum(item["incomplete_key_suffix"] for item in all_window_outputs),
        "rejection_reason_counts": dict(
            sorted(Counter(reason for item in all_window_outputs for reason in item["reasons"]).items())
        ),
    }

    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in result["samples"]:
        by_k[item["k"]].append(item)
    result["summary"] = {str(k): summarize(by_k[k]) for k in ks}
    result["status"] = "pass"
    write_json(run_dir / "streaming_text.json", result)
    write_json(args.output_dir / "streaming_text_latest.json", result)
    print(json.dumps({"status": result["status"], "scope": result["scope"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
