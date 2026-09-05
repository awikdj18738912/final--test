#!/usr/bin/env python3
"""Paced 1/2/4-session WebSocket load test for a running Gate1 service."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
import websockets


ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = Path(
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0768/BAC009S0768W0452.wav"
)


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    point = (len(ordered) - 1) * value
    lower = math.floor(point)
    upper = math.ceil(point)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


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


def load_pcm(audio_path: Path, seconds: float) -> bytes:
    waveform, sample_rate = sf.read(audio_path, always_2d=False)
    waveform = np.asarray(waveform)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError("load test input must be 16 kHz")
    sample_count = min(len(waveform), int(round(seconds * sample_rate)))
    if sample_count < sample_rate:
        raise ValueError("load test needs at least one second of audio")
    return np.clip(waveform[:sample_count] * 32768, -32768, 32767).astype("<i2").tobytes()


async def run_session(
    client: httpx.AsyncClient,
    base_url: str,
    tenant_id: str,
    pcm: bytes,
    frame_ms: int,
) -> dict[str, Any]:
    response = await client.post(
        "/sessions",
        headers={"Idempotency-Key": f"load-{uuid.uuid4().hex}"},
        json={"tenant_id": tenant_id, "mode": "realtime", "language": "Chinese", "quality_level": "balanced"},
    )
    response.raise_for_status()
    session = response.json()
    session_id = session["session_id"]
    websocket_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    websocket_url += f"/sessions/{session_id}/stream?tenant_id={tenant_id}"
    first_result_sec: float | None = None
    result_version = 0
    start = time.perf_counter()
    frame_bytes = int(16000 * frame_ms / 1000) * 2
    try:
        async with websockets.connect(websocket_url, max_size=2**20) as socket:
            ready = json.loads(await socket.recv())
            if ready.get("event") != "ready":
                raise RuntimeError(f"unexpected ready event: {ready}")

            async def receive_events() -> dict[str, Any]:
                nonlocal first_result_sec, result_version
                while True:
                    event = json.loads(await socket.recv())
                    if event.get("event") in {"partial", "final"}:
                        result_version = int(event["result_version"])
                        if first_result_sec is None and event.get("text"):
                            first_result_sec = time.perf_counter() - start
                    if event.get("event") == "final":
                        return event
                    if event.get("event") == "error":
                        raise RuntimeError(event.get("detail", "websocket error"))

            receiver = asyncio.create_task(receive_events())
            try:
                for offset in range(0, len(pcm), frame_bytes):
                    scheduled = offset / 2 / 16000
                    remaining = scheduled - (time.perf_counter() - start)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await socket.send(pcm[offset : offset + frame_bytes])
                await socket.send(json.dumps({"event": "end"}))
                final = await asyncio.wait_for(receiver, timeout=60)
            finally:
                if not receiver.done():
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver
        return {
            "session_id": session_id,
            "status": "pass",
            "first_result_sec": first_result_sec,
            "final_result_sec": time.perf_counter() - start,
            "result_version": result_version,
            "final_text": final.get("text", ""),
        }
    except Exception as exc:
        return {"session_id": session_id, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.suppress(Exception):
            await client.delete(f"/sessions/{session_id}", params={"tenant_id": tenant_id})


async def run(args: argparse.Namespace) -> dict[str, Any]:
    pcm = load_pcm(args.audio, args.audio_sec)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        health = (await client.get("/health")).json()
        if health.get("status") != "ok":
            raise RuntimeError(f"service is not healthy: {health}")
        for concurrency in args.concurrency:
            for repeat in range(args.repetitions):
                tenant_id = f"load-c{concurrency}-r{repeat + 1}"
                sample = await asyncio.gather(
                    *(run_session(client, args.url, tenant_id, pcm, args.frame_ms) for _ in range(concurrency))
                )
                results.append({"concurrency": concurrency, "repeat": repeat + 1, "sessions": sample})

    tiers: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        sessions = [item for round_result in results if round_result["concurrency"] == concurrency for item in round_result["sessions"]]
        first = [item["first_result_sec"] for item in sessions if item.get("first_result_sec") is not None]
        final = [item["final_result_sec"] for item in sessions if item.get("final_result_sec") is not None]
        tiers.append(
            {
                "concurrency": concurrency,
                "pass_count": sum(item["status"] == "pass" for item in sessions),
                "total_count": len(sessions),
                "first_result_sec": summarize(first),
                "final_result_sec": summarize(final),
            }
        )
    successful = all(tier["pass_count"] == tier["total_count"] for tier in tiers)
    return {
        "benchmark": "Gate 1 realtime WebSocket paced load",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if successful else "fail",
        "service_url": args.url,
        "service_health_before_run": health,
        "audio": str(args.audio),
        "audio_sec_per_session": args.audio_sec,
        "frame_ms": args.frame_ms,
        "repetitions": args.repetitions,
        "tiers": tiers,
        "rounds": results,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--audio-sec", type=float, default=5.0)
    parser.add_argument("--frame-ms", type=int, choices=[20, 40, 100, 250], default=250)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    write_json(run_dir / "load_test.json", result)
    write_json(args.output_dir / "load_test_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(tier["pass_count"] == tier["total_count"] for tier in result["tiers"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
