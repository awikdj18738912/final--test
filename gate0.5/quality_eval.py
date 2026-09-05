#!/usr/bin/env python3
"""Evaluate one streaming ASR backend on the shared Gate 0.5 manifest.

Run this file from the Conda environment belonging to the selected backend.
Quality decoding is buffered rather than real-time paced; latency SLO evidence
remains in the separate 1/2/4-session benchmarks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "quality_manifest_200.json"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "p50": round(percentile(values, 0.50), 6) if values else None,
        "p95": round(percentile(values, 0.95), 6) if values else None,
        "p99": round(percentile(values, 0.99), 6) if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


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


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != 16000:
        import torch
        import torchaudio.functional as audio_functional

        waveform = audio_functional.resample(torch.from_numpy(waveform), sample_rate, 16000).numpy()
        sample_rate = 16000
    return np.ascontiguousarray(waveform), sample_rate


def output_text(result: Any) -> str:
    if not result:
        return ""
    if isinstance(result, dict):
        result = [result]
    return "".join(str(item.get("text", "")) for item in result if isinstance(item, dict))


class Backend(Protocol):
    identity: dict[str, Any]

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]: ...


class QwenStreamingBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        from qwen_asr import Qwen3ASRModel

        started = time.perf_counter()
        self.model = Qwen3ASRModel.LLM(
            model=str(args.model.resolve()),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_inference_batch_size=1,
            max_new_tokens=args.max_new_tokens,
        )
        self.chunk_sec = args.qwen_chunk_sec
        self.push_sec = args.qwen_push_sec
        self.identity = {
            "backend": "qwen-streaming",
            "model_path": str(args.model.resolve()),
            "load_sec": round(time.perf_counter() - started, 4),
            "chunk_sec": self.chunk_sec,
            "push_sec": self.push_sec,
            "qwen_asr": package_version("qwen-asr"),
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
        }

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
        state = self.model.init_streaming_state(language="Chinese", chunk_size_sec=self.chunk_sec)
        stride = max(1, int(round(self.push_sec * sample_rate)))
        started = time.perf_counter()
        first_text_sec = None
        input_at_first_text_sec = None
        for offset in range(0, len(waveform), stride):
            chunk = waveform[offset : offset + stride]
            self.model.streaming_transcribe(chunk, state)
            if first_text_sec is None and state.text:
                first_text_sec = time.perf_counter() - started
                input_at_first_text_sec = min((offset + len(chunk)) / sample_rate, len(waveform) / sample_rate)
        self.model.finish_streaming_transcribe(state)
        elapsed = time.perf_counter() - started
        if first_text_sec is None and state.text:
            first_text_sec = elapsed
            input_at_first_text_sec = len(waveform) / sample_rate
        return {"text": state.text, "elapsed_sec": elapsed, "compute_ttfp_sec": first_text_sec, "input_audio_at_first_text_sec": input_at_first_text_sec}


class ParaformerOnlineBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        from funasr import AutoModel

        started = time.perf_counter()
        self.model = AutoModel(model=str(args.model.resolve()), device=f"cuda:{args.gpu}", disable_update=True)
        self.chunk_sec = args.para_chunk_sec
        self.chunk_size = args.para_chunk_size
        self.encoder_look_back = args.encoder_look_back
        self.decoder_look_back = args.decoder_look_back
        self.identity = {
            "backend": "paraformer-online",
            "model_path": str(args.model.resolve()),
            "load_sec": round(time.perf_counter() - started, 4),
            "chunk_sec": self.chunk_sec,
            "chunk_size": self.chunk_size,
            "encoder_chunk_look_back": self.encoder_look_back,
            "decoder_chunk_look_back": self.decoder_look_back,
            "funasr": package_version("funasr"),
            "modelscope": package_version("modelscope"),
            "torch": package_version("torch"),
        }

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        stride = max(1, int(round(self.chunk_sec * sample_rate)))
        started = time.perf_counter()
        first_text_sec = None
        input_at_first_text_sec = None
        parts: list[str] = []
        for offset in range(0, len(waveform), stride):
            chunk = waveform[offset : offset + stride]
            result = self.model.generate(
                input=chunk,
                cache=cache,
                is_final=offset + stride >= len(waveform),
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_look_back,
                decoder_chunk_look_back=self.decoder_look_back,
            )
            text = output_text(result)
            if text:
                parts.append(text)
                if first_text_sec is None:
                    first_text_sec = time.perf_counter() - started
                    input_at_first_text_sec = min((offset + len(chunk)) / sample_rate, len(waveform) / sample_rate)
        return {
            "text": "".join(parts),
            "elapsed_sec": time.perf_counter() - started,
            "compute_ttfp_sec": first_text_sec,
            "input_audio_at_first_text_sec": input_at_first_text_sec,
        }


def metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [item for item in samples if item["status"] == "pass"]
    reference_chars = sum(item["reference_chars"] for item in passed)
    errors = sum(item["edit_distance"] for item in passed)
    macro_cer = [item["cer"] for item in passed]
    return {
        "samples_passed": len(passed),
        "samples_total": len(samples),
        "success_rate": round(len(passed) / len(samples), 6) if samples else None,
        "reference_chars": reference_chars,
        "edit_distance": errors,
        "corpus_cer": round(errors / reference_chars, 6) if reference_chars else None,
        "macro_cer": summarize(macro_cer),
        "compute_rtf": summarize([item["compute_rtf"] for item in passed]),
        "elapsed_sec": summarize([item["elapsed_sec"] for item in passed]),
        "audio_sec": round(sum(item["audio_sec"] for item in passed), 4),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    backend: Backend = QwenStreamingBackend(args) if args.backend == "qwen-streaming" else ParaformerOnlineBackend(args)

    warm_waveform, warm_rate = load_audio(Path(selected[0]["audio_filepath"]))
    warm_started = time.perf_counter()
    warmup = backend.transcribe(warm_waveform, warm_rate)
    warmup["wall_sec"] = round(time.perf_counter() - warm_started, 4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.backend / timestamp
    result: dict[str, Any] = {
        "benchmark": "Gate 0.5 stratified streaming quality evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "backend": backend.identity,
        "environment": {"python": os.sys.version},
        "manifest": str(args.manifest.resolve()),
        "manifest_source_sha256": manifest["source_sha256"],
        "requested_samples": len(selected),
        "warmup": warmup,
        "samples": [],
    }
    for index, item in enumerate(selected, start=1):
        sample_started = time.perf_counter()
        record = {
            "index": index,
            "utterance_id": item["utterance_id"],
            "speaker": item["speaker"],
            "stratum": item["stratum"],
            "audio_filepath": item["audio_filepath"],
            "manifest_duration_sec": item["duration"],
            "reference": item["text"],
        }
        try:
            waveform, sample_rate = load_audio(Path(item["audio_filepath"]))
            decoded = backend.transcribe(waveform, sample_rate)
            reference = normalize(item["text"])
            hypothesis = normalize(decoded["text"])
            distance = edit_distance(reference, hypothesis)
            audio_sec = len(waveform) / sample_rate
            record.update(
                decoded,
                status="pass" if decoded["text"] else "fail",
                normalized_reference=reference,
                normalized_hypothesis=hypothesis,
                reference_chars=len(reference),
                edit_distance=distance,
                cer=distance / len(reference) if reference else None,
                audio_sec=audio_sec,
                compute_rtf=decoded["elapsed_sec"] / audio_sec,
            )
        except Exception as exc:
            record.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        record["sample_wall_sec"] = time.perf_counter() - sample_started
        result["samples"].append(record)
        if index % args.checkpoint_every == 0:
            result["progress"] = {"completed": index, "total": len(selected)}
            write_json(run_dir / "quality_eval.partial.json", result)

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in result["samples"]:
        by_stratum[item["stratum"]].append(item)
        by_speaker[item["speaker"]].append(item)
    result["summary"] = metrics(result["samples"])
    result["summary"]["by_stratum"] = {name: metrics(items) for name, items in sorted(by_stratum.items())}
    result["summary"]["by_speaker"] = {name: metrics(items) for name, items in sorted(by_speaker.items())}
    result["status"] = "pass" if result["summary"]["samples_passed"] == len(selected) else "fail"
    result["progress"] = {"completed": len(selected), "total": len(selected)}
    write_json(run_dir / "quality_eval.json", result)
    write_json(args.output_dir / f"{args.backend}_latest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen-streaming", "paraformer-online"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "quality")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-chunk-sec", type=float, default=1.0)
    parser.add_argument("--qwen-push-sec", type=float, default=0.25)
    parser.add_argument("--para-chunk-sec", type=float, default=0.6)
    parser.add_argument("--para-chunk-size", type=int, nargs=3, default=[0, 10, 5])
    parser.add_argument("--encoder-look-back", type=int, default=4)
    parser.add_argument("--decoder-look-back", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.checkpoint_every <= 0:
        parser.error("checkpoint-every must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "backend": result["backend"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
