#!/usr/bin/env python3
"""Warm capacity benchmark for Qwen3-ASR's official streaming state API.

Fast runs feed buffered audio immediately and measure decoder compute RTF.  A
separate paced probe feeds frames according to their audio timestamps and
stops at the first non-empty hypothesis, yielding an end-to-end client-visible
TTFP.  Keeping them separate prevents audio accumulation time from being
mistaken for model compute time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_gate0 import MemorySampler, load_waveform, write_json


ROOT = Path(__file__).resolve().parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * value
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": round(percentile(values, 0.50), 4) if values else None,
        "p95": round(percentile(values, 0.95), 4) if values else None,
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
        "mean": round(statistics.fmean(values), 4) if values else None,
    }


def fast_complete_run(
    model: Any,
    waveform: np.ndarray,
    sample_rate: int,
    language: str,
    chunk_size_sec: float,
    input_push_sec: float,
) -> dict[str, Any]:
    state = model.init_streaming_state(language=language, chunk_size_sec=chunk_size_sec)
    push_samples = max(1, int(round(input_push_sec * sample_rate)))
    start = time.perf_counter()
    first_text_at: float | None = None
    input_audio_at_first_text: float | None = None
    for offset in range(0, len(waveform), push_samples):
        model.streaming_transcribe(waveform[offset : offset + push_samples], state)
        if first_text_at is None and state.text:
            first_text_at = time.perf_counter() - start
            input_audio_at_first_text = min((offset + push_samples) / sample_rate, len(waveform) / sample_rate)
    model.finish_streaming_transcribe(state)
    elapsed = time.perf_counter() - start
    if first_text_at is None and state.text:
        first_text_at = elapsed
        input_audio_at_first_text = len(waveform) / sample_rate
    audio_sec = len(waveform) / sample_rate
    return {
        "status": "pass" if state.text else "fail",
        "text": state.text,
        "elapsed_sec": elapsed,
        "compute_ttfp_sec": first_text_at,
        "input_audio_at_first_text_sec": input_audio_at_first_text,
        "compute_rtf": elapsed / audio_sec if audio_sec else None,
    }


def paced_ttfp_probe(
    model: Any,
    waveform: np.ndarray,
    sample_rate: int,
    language: str,
    chunk_size_sec: float,
    input_push_sec: float,
) -> dict[str, Any]:
    state = model.init_streaming_state(language=language, chunk_size_sec=chunk_size_sec)
    push_samples = max(1, int(round(input_push_sec * sample_rate)))
    start = time.perf_counter()
    for offset in range(0, len(waveform), push_samples):
        scheduled = offset / sample_rate
        remaining = scheduled - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
        model.streaming_transcribe(waveform[offset : offset + push_samples], state)
        if state.text:
            return {
                "status": "pass",
                "e2e_ttfp_sec": time.perf_counter() - start,
                "input_audio_at_first_text_sec": min((offset + push_samples) / sample_rate, len(waveform) / sample_rate),
                "text": state.text,
            }
    model.finish_streaming_transcribe(state)
    return {
        "status": "pass" if state.text else "fail",
        "e2e_ttfp_sec": time.perf_counter() - start if state.text else None,
        "input_audio_at_first_text_sec": len(waveform) / sample_rate if state.text else None,
        "text": state.text,
    }


def benchmark_variant(
    model: Any,
    model_id: str,
    variant: dict[str, Any],
    waveform: np.ndarray,
    sample_rate: int,
    matrix: dict[str, Any],
) -> dict[str, Any]:
    chunk_size = float(variant["chunk_size_sec"])
    push_size = float(variant["input_push_sec"])
    language = str(matrix["language"])

    # Compiled kernels and model-side caches must not inflate the warm samples.
    warmup = fast_complete_run(model, waveform, sample_rate, language, chunk_size, push_size)
    samples: list[dict[str, Any]] = []
    sampler = MemorySampler(int(matrix["gpu"]))
    sampler.start()
    try:
        for index in range(int(matrix["repetitions"])):
            fast = fast_complete_run(model, waveform, sample_rate, language, chunk_size, push_size)
            paced = paced_ttfp_probe(model, waveform, sample_rate, language, chunk_size, push_size)
            samples.append({"iteration": index + 1, "fast": fast, "paced": paced})
    finally:
        sampler.stop()

    compute_rtf = [item["fast"]["compute_rtf"] for item in samples if item["fast"]["compute_rtf"] is not None]
    compute_ttfp = [item["fast"]["compute_ttfp_sec"] for item in samples if item["fast"]["compute_ttfp_sec"] is not None]
    e2e_ttfp = [item["paced"]["e2e_ttfp_sec"] for item in samples if item["paced"]["e2e_ttfp_sec"] is not None]
    result = {
        "model_id": model_id,
        "variant": variant,
        "warmup": warmup,
        "samples": samples,
        "summary": {
            "compute_rtf": summarize(compute_rtf),
            "compute_ttfp_sec": summarize(compute_ttfp),
            "e2e_ttfp_sec": summarize(e2e_ttfp),
            "memory": sampler.summary(),
        },
    }
    target = matrix["acceptance"]
    result["meets_target"] = (
        result["summary"]["compute_rtf"]["p95"] is not None
        and result["summary"]["e2e_ttfp_sec"]["p95"] is not None
        and result["summary"]["compute_rtf"]["p95"] < float(target["compute_rtf_p95"])
        and result["summary"]["e2e_ttfp_sec"]["p95"] <= float(target["realtime_ttfp_p95_sec"])
    )
    return result


def run(matrix: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from qwen_asr import Qwen3ASRModel

    audio_path = Path(matrix["audio"])
    waveform, sample_rate = load_waveform(audio_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for model_spec in matrix["models"]:
        model_path = Path(model_spec["path"])
        load_start = time.perf_counter()
        model = Qwen3ASRModel.LLM(
            model=str(model_path),
            gpu_memory_utilization=float(matrix["gpu_memory_utilization"]),
            max_model_len=int(matrix["max_model_len"]),
            max_inference_batch_size=1,
            max_new_tokens=int(matrix["max_new_tokens"]),
        )
        model_record: dict[str, Any] = {
            "id": model_spec["id"],
            "path": str(model_path),
            "load_sec": round(time.perf_counter() - load_start, 4),
            "variants": [],
        }
        for variant in matrix["variants"]:
            model_record["variants"].append(
                benchmark_variant(model, str(model_spec["id"]), variant, waveform, sample_rate, matrix)
            )
        records.append(model_record)
        del model

    matrix_copy = dict(matrix)
    matrix_copy["audio_sec"] = round(len(waveform) / sample_rate, 4)
    results = [variant for model in records for variant in model["variants"]]
    aggregate = {
        "benchmark": "Gate 0 warm streaming capacity",
        "timestamp": now_iso(),
        "status": "pass" if any(result["meets_target"] for result in results) else "capacity_boundary",
        "metric_definition": {
            "compute_rtf": "Fast buffered-input elapsed time divided by audio duration; measures decode throughput only.",
            "compute_ttfp_sec": "Time to first text when buffered audio is delivered immediately; diagnostic only.",
            "e2e_ttfp_sec": "Time to first text while input frames are paced by their audio timestamps; client-visible single-session latency.",
        },
        "matrix": matrix_copy,
        "models": records,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "capacity.json", aggregate)
    write_json(output_dir / "capacity_latest.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "capacity_matrix.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "capacity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    # The benchmark is a GPU0-only experiment and must not share the card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(matrix["gpu"])
    result = run(matrix, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
