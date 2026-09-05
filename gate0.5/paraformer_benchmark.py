#!/usr/bin/env python3
"""Gate 0.5 benchmark for FunASR Paraformer online decoding.

The benchmark deliberately separates buffered-input throughput from a paced
client simulation.  Paraformer is a stateful online recognizer, so every
session owns an independent ``cache`` dictionary while all sessions share one
loaded model instance.  The output is written as JSON evidence rather than
being used to make an automatic backend-selection claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_AUDIO = Path(
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0913/BAC009S0913W0321.wav"
)
DEFAULT_TRANSCRIPT = Path(
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt"
)
DEFAULT_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": round(percentile(values, 0.50), 4) if values else None,
        "p95": round(percentile(values, 0.95), 4) if values else None,
        "p99": round(percentile(values, 0.99), 4) if values else None,
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
        "mean": round(statistics.fmean(values), 4) if values else None,
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


class MemorySampler:
    """Sample aggregate GPU memory so model load and inference share one peak."""

    def __init__(self, gpu: int, interval_sec: float = 0.2) -> None:
        self.gpu = gpu
        self.interval_sec = interval_sec
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> None:
        result = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if result.returncode:
            return
        for line in result.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 3 and fields[0] == str(self.gpu):
                try:
                    self.samples.append(int(fields[1]))
                except ValueError:
                    pass

    def start(self) -> None:
        self.sample()

        def loop() -> None:
            while not self.stop_event.is_set():
                self.sample()
                self.stop_event.wait(self.interval_sec)

        self.thread = threading.Thread(target=loop, name=f"gpu{self.gpu}-memory", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.sample()

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"peak_used_mib": None, "peak_used_gib": None, "samples": 0}
        peak = max(self.samples)
        return {"peak_used_mib": peak, "peak_used_gib": round(peak / 1024, 3), "samples": len(self.samples)}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    waveform, sample_rate = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != 16000:
        import torchaudio.functional as audio_functional
        import torch

        waveform = audio_functional.resample(torch.from_numpy(waveform), sample_rate, 16000).numpy()
        sample_rate = 16000
    return np.ascontiguousarray(waveform), sample_rate


def read_reference(transcript: Path, utterance_id: str) -> str | None:
    if not transcript.exists():
        return None
    with transcript.open(encoding="utf-8") as source:
        for line in source:
            key, separator, value = line.rstrip("\n").partition(" ")
            if key == utterance_id and separator:
                return value
    return None


def normalize_for_cer(value: str) -> str:
    return re.sub(r"[\W_]", "", value, flags=re.UNICODE).lower()


def character_error_rate(reference: str | None, hypothesis: str) -> float | None:
    if reference is None:
        return None
    reference, hypothesis = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
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
    return previous[-1] / len(reference)


def output_text(result: Any) -> str:
    """FunASR returns a list of result dictionaries for a single input."""
    if not result:
        return ""
    if isinstance(result, dict):
        result = [result]
    return "".join(str(item.get("text", "")) for item in result if isinstance(item, dict))


def stream_once(
    model: Any,
    waveform: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    paced: bool,
    chunk_size: list[int],
    encoder_look_back: int,
    decoder_look_back: int,
) -> dict[str, Any]:
    """Run one independent online stream and return client and compute timing."""
    cache: dict[str, Any] = {}
    stride = max(1, int(round(chunk_sec * sample_rate)))
    start = time.perf_counter()
    first_text_at: float | None = None
    input_at_first_text: float | None = None
    parts: list[str] = []
    try:
        for offset in range(0, len(waveform), stride):
            if paced:
                deadline = offset / sample_rate
                sleep_for = deadline - (time.perf_counter() - start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            chunk = waveform[offset : offset + stride]
            is_final = offset + stride >= len(waveform)
            generated = model.generate(
                input=chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_look_back,
                decoder_chunk_look_back=decoder_look_back,
            )
            text = output_text(generated)
            if text:
                parts.append(text)
                if first_text_at is None:
                    first_text_at = time.perf_counter() - start
                    input_at_first_text = min((offset + len(chunk)) / sample_rate, len(waveform) / sample_rate)
        elapsed = time.perf_counter() - start
        transcript = "".join(parts)
        return {
            "status": "pass" if transcript else "fail",
            "text": transcript,
            "elapsed_sec": elapsed,
            "first_text_sec": first_text_at,
            "input_audio_at_first_text_sec": input_at_first_text,
            "rtf": elapsed / (len(waveform) / sample_rate),
        }
    except Exception as exc:
        return {"status": "fail", "error": f"{type(exc).__name__}: {exc}", "text": ""}


def parallel_probe(
    model: Any,
    waveform: np.ndarray,
    sample_rate: int,
    concurrency: int,
    chunk_sec: float,
    chunk_size: list[int],
    encoder_look_back: int,
    decoder_look_back: int,
) -> dict[str, Any]:
    """Inject simultaneous sessions through one serialized CUDA model.

    A single FunASR ``AutoModel`` is shared by design.  Calling it from several
    Python threads would race its module state and does not represent a
    production single-worker scheduler.  Each loop iteration instead delivers
    the same real-time audio frame to every session in round-robin order.
    """
    start = time.perf_counter()
    stride = max(1, int(round(chunk_sec * sample_rate)))
    streams: list[dict[str, Any]] = [
        {"cache": {}, "parts": [], "first_text_sec": None, "input_audio_at_first_text_sec": None}
        for _ in range(concurrency)
    ]
    try:
        for offset in range(0, len(waveform), stride):
            deadline = offset / sample_rate
            sleep_for = deadline - (time.perf_counter() - start)
            if sleep_for > 0:
                time.sleep(sleep_for)
            chunk = waveform[offset : offset + stride]
            is_final = offset + stride >= len(waveform)
            for session in streams:
                generated = model.generate(
                    input=chunk,
                    cache=session["cache"],
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_look_back,
                    decoder_chunk_look_back=decoder_look_back,
                )
                text = output_text(generated)
                if text:
                    session["parts"].append(text)
                    if session["first_text_sec"] is None:
                        session["first_text_sec"] = time.perf_counter() - start
                        session["input_audio_at_first_text_sec"] = min(
                            (offset + len(chunk)) / sample_rate, len(waveform) / sample_rate
                        )
        end = time.perf_counter() - start
        for session in streams:
            session["text"] = "".join(session.pop("parts"))
            session.pop("cache")
            session["elapsed_sec"] = end
            session["rtf"] = end / (len(waveform) / sample_rate)
            session["status"] = "pass" if session["text"] else "fail"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for session in streams:
            session.pop("cache", None)
            session.pop("parts", None)
            session["status"] = "fail"
            session["error"] = message
            session.setdefault("text", "")
    elapsed = time.perf_counter() - start
    first_text = [item["first_text_sec"] for item in streams if item.get("first_text_sec") is not None]
    finals = [item["elapsed_sec"] for item in streams if item.get("elapsed_sec") is not None]
    return {
        "concurrency": concurrency,
        "status": "pass" if all(item["status"] == "pass" for item in streams) else "fail",
        "wall_elapsed_sec": elapsed,
        "streams": streams,
        "summary": {"ttfp_sec": summarize(first_text), "end_to_final_sec": summarize(finals)},
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Import after argparse so '--help' works before the optional benchmark stack is installed.
    from funasr import AutoModel
    from modelscope.hub.snapshot_download import snapshot_download

    audio_path = args.audio.resolve()
    waveform, sample_rate = load_audio(audio_path)
    if args.max_audio_sec is not None:
        waveform = waveform[: int(round(args.max_audio_sec * sample_rate))]
    reference = args.reference_text
    reference_source = "--reference-text"
    if reference is None:
        reference = read_reference(args.transcript, audio_path.stem)
        reference_source = str(args.transcript.resolve()) if reference is not None else None
    model_path = args.model_path
    if model_path is None:
        model_path = Path(snapshot_download(args.model, revision=args.revision, cache_dir=str(args.model_cache)))
    else:
        model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Paraformer model path does not exist: {model_path}")

    sampler = MemorySampler(args.gpu)
    sampler.start()
    try:
        load_start = time.perf_counter()
        model = AutoModel(model=str(model_path), device=f"cuda:{args.gpu}", disable_update=True)
        load_sec = time.perf_counter() - load_start

        # Warm compiled kernels and model caches before evidence samples.
        warmup = stream_once(
            model,
            waveform,
            sample_rate,
            args.chunk_sec,
            False,
            args.chunk_size,
            args.encoder_look_back,
            args.decoder_look_back,
        )
        buffered_samples = [
            stream_once(
                model,
                waveform,
                sample_rate,
                args.chunk_sec,
                False,
                args.chunk_size,
                args.encoder_look_back,
                args.decoder_look_back,
            )
            for _ in range(args.repetitions)
        ]
        paced_samples = [
            stream_once(
                model,
                waveform,
                sample_rate,
                args.chunk_sec,
                True,
                args.chunk_size,
                args.encoder_look_back,
                args.decoder_look_back,
            )
            for _ in range(args.repetitions)
        ]
        concurrency_samples = []
        for level in args.concurrency:
            for repeat in range(1, args.repetitions + 1):
                sample = parallel_probe(
                    model,
                    waveform,
                    sample_rate,
                    level,
                    args.chunk_sec,
                    args.chunk_size,
                    args.encoder_look_back,
                    args.decoder_look_back,
                )
                sample["repeat"] = repeat
                concurrency_samples.append(sample)
    finally:
        sampler.stop()

    buffered_rtf = [item["rtf"] for item in buffered_samples if item.get("rtf") is not None]
    buffered_ttfp = [item["first_text_sec"] for item in buffered_samples if item.get("first_text_sec") is not None]
    paced_ttfp = [item["first_text_sec"] for item in paced_samples if item.get("first_text_sec") is not None]
    paced_final = [item["elapsed_sec"] for item in paced_samples if item.get("elapsed_sec") is not None]
    cer = [character_error_rate(reference, item.get("text", "")) for item in paced_samples]
    cer = [item for item in cer if item is not None]
    concurrency_summary = []
    for level in args.concurrency:
        runs = [item for item in concurrency_samples if item["concurrency"] == level]
        streams = [stream for item in runs for stream in item["streams"]]
        ttfp = [item["first_text_sec"] for item in streams if item.get("first_text_sec") is not None]
        final = [item["elapsed_sec"] for item in streams if item.get("elapsed_sec") is not None]
        stream_cer = [character_error_rate(reference, item.get("text", "")) for item in streams]
        stream_cer = [item for item in stream_cer if item is not None]
        passed_streams = sum(item.get("status") == "pass" for item in streams)
        concurrency_summary.append(
            {
                "concurrency": level,
                "runs_passed": sum(item["status"] == "pass" for item in runs),
                "runs_total": len(runs),
                "streams_passed": passed_streams,
                "streams_total": len(streams),
                "success_rate": round(passed_streams / len(streams), 4) if streams else None,
                "ttfp_sec": summarize(ttfp),
                "end_to_final_sec": summarize(final),
                "cer": summarize(stream_cer),
            }
        )
    successful = all(item["status"] == "pass" for item in [warmup, *buffered_samples, *paced_samples, *concurrency_samples])
    return {
        "benchmark": "Gate 0.5 FunASR Paraformer online",
        "timestamp": now_iso(),
        "status": "pass" if successful else "fail",
        "backend": {
            "implementation": "FunASR AutoModel online cache API",
            "model_id": args.model,
            "model_path": str(model_path),
            "revision": args.revision,
            "device": f"cuda:{args.gpu}",
            "chunk_size": args.chunk_size,
            "input_chunk_sec": args.chunk_sec,
            "encoder_chunk_look_back": args.encoder_look_back,
            "decoder_chunk_look_back": args.decoder_look_back,
        },
        "environment": {
            "funasr": package_version("funasr"),
            "modelscope": package_version("modelscope"),
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "python": os.sys.version,
        },
        "audio": {
            "path": str(audio_path),
            "utterance_id": audio_path.stem,
            "duration_sec": round(len(waveform) / sample_rate, 4),
            "sample_rate": sample_rate,
            "reference": reference,
            "reference_source": reference_source,
            "max_audio_sec": args.max_audio_sec,
        },
        "load_sec": round(load_sec, 4),
        "warmup": warmup,
        "buffered_samples": buffered_samples,
        "paced_samples": paced_samples,
        "concurrency_samples": concurrency_samples,
        "concurrency_summary": concurrency_summary,
        "summary": {
            "compute_rtf": summarize(buffered_rtf),
            "buffered_ttfp_sec": summarize(buffered_ttfp),
            "e2e_ttfp_sec": summarize(paced_ttfp),
            "e2e_end_to_final_sec": summarize(paced_final),
            "cer": summarize(cer),
            "memory": sampler.summary(),
            "metric_definition": {
                "compute_rtf": "Buffered-input elapsed time divided by audio duration; decoder throughput only.",
                "e2e_ttfp_sec": "First non-empty text while audio chunks arrive at real-time timestamps.",
                "e2e_end_to_final_sec": "Paced input start to final online result; this is not an offline 2-pass result.",
                "cer": "Character error rate after removing whitespace and punctuation from AISHELL reference and hypothesis.",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--reference-text", help="Override the transcript reference, for example for a clipped input.")
    parser.add_argument("--max-audio-sec", type=float, help="Decode only the first N seconds of the input.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path, help="Use an already downloaded Paraformer directory.")
    parser.add_argument("--model-cache", type=Path, default=Path("/home/aim0/data/models/funasr"))
    parser.add_argument("--revision", default="master")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chunk-sec", type=float, default=0.6)
    parser.add_argument("--chunk-size", type=int, nargs=3, default=[0, 10, 5])
    parser.add_argument("--encoder-look-back", type=int, default=4)
    parser.add_argument("--decoder-look-back", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4", help="Comma-separated simultaneous online sessions.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.concurrency = [int(value) for value in args.concurrency.split(",") if value.strip()]
    if (
        args.chunk_sec <= 0
        or args.repetitions <= 0
        or not args.concurrency
        or min(args.concurrency) <= 0
        or (args.max_audio_sec is not None and args.max_audio_sec <= 0)
    ):
        parser.error("chunk-sec, max-audio-sec, repetitions, and every concurrency level must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = run(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    write_json(run_dir / "paraformer_online.json", result)
    write_json(args.output_dir / "paraformer_online_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
