#!/usr/bin/env python3
"""One-session paced audio smoke for Gate1 + asynchronous Gate2 K=1."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
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
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/"
    "S0768/BAC009S0768W0452.wav"
)


def load_pcm(path: Path, seconds: float | None) -> bytes:
    waveform, sample_rate = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError("smoke audio must be 16 kHz")
    if seconds is not None:
        waveform = waveform[: int(round(seconds * sample_rate))]
    return np.clip(waveform * 32768, -32768, 32767).astype("<i2").tobytes()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    pcm = load_pcm(args.audio, args.audio_sec)
    tenant_id = f"live-smoke-{uuid.uuid4().hex[:8]}"
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        health = (await client.get("/health")).json()
        if health.get("status") != "ok" or health.get("gpu1_role") != "refiner":
            raise RuntimeError(f"service is not in refiner mode: {health}")
        response = await client.post(
            "/sessions",
            headers={"Idempotency-Key": f"live-{uuid.uuid4().hex}"},
            json={"tenant_id": tenant_id, "mode": "realtime", "language": "Chinese"},
        )
        response.raise_for_status()
        session_id = response.json()["session_id"]
        websocket_url = args.url.replace("http://", "ws://").replace("https://", "wss://")
        websocket_url += f"/sessions/{session_id}/stream?tenant_id={tenant_id}"
        started = time.perf_counter()
        frame_bytes = int(16000 * args.frame_ms / 1000) * 2
        try:
            async with websockets.connect(websocket_url, max_size=2**22) as socket:
                ready = json.loads(await socket.recv())
                ready["received_sec"] = round(time.perf_counter() - started, 6)
                events.append(ready)

                async def receive_until_complete() -> None:
                    while True:
                        event = json.loads(await socket.recv())
                        event["received_sec"] = round(time.perf_counter() - started, 6)
                        events.append(event)
                        if event.get("event") == "error":
                            raise RuntimeError(event.get("detail", "service error"))
                        if event.get("event") == "complete":
                            return

                receiver = asyncio.create_task(receive_until_complete())
                try:
                    for offset in range(0, len(pcm), frame_bytes):
                        scheduled = offset / 2 / 16000
                        remaining = scheduled - (time.perf_counter() - started)
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                        await socket.send(pcm[offset : offset + frame_bytes])
                    await socket.send(json.dumps({"event": "end"}))
                    await asyncio.wait_for(receiver, timeout=args.timeout)
                finally:
                    if not receiver.done():
                        receiver.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await receiver
            snapshot = (await client.get(f"/sessions/{session_id}/result", params={"tenant_id": tenant_id})).json()
        finally:
            with contextlib.suppress(Exception):
                await client.delete(f"/sessions/{session_id}", params={"tenant_id": tenant_id})

    final_events = [item for item in events if item["event"] == "final"]
    refinements = [item for item in events if item["event"].startswith("refiner_") or item["event"] == "revision"]
    revisions = [item for item in refinements if item["event"] == "revision"]
    complete = next(item for item in events if item["event"] == "complete")
    return {
        "benchmark": "Gate1 + Gate2 K1 real-audio asynchronous integration smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if final_events and refinements else "fail",
        "service": args.url,
        "health": health,
        "audio": str(args.audio.resolve()),
        "audio_sec": len(pcm) / 2 / 16000,
        "frame_ms": args.frame_ms,
        "raw_final_visible_sec": final_events[0]["received_sec"] if final_events else None,
        "complete_sec": complete["received_sec"],
        "refinement_events": len(refinements),
        "revision_events": len(revisions),
        "final_raw_text": final_events[-1].get("raw_text") if final_events else None,
        "final_visible_text": complete["text"],
        "session_snapshot": snapshot,
        "events": events,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--audio-sec", type=float, default=5.0)
    parser.add_argument("--frame-ms", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "live_integration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "live_smoke.json", result)
    write_json(args.output_dir / "live_smoke_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
