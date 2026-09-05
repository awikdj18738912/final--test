#!/usr/bin/env python3
"""Run paced 1/2/4-session Gate1 + Gate2 real-audio concurrency steps."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .live_smoke import DEFAULT_AUDIO, run as run_session, write_json


ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return round(ordered[index], 6)


async def run_level(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    session_args = SimpleNamespace(
        url=args.url,
        audio=args.audio,
        audio_sec=args.audio_sec,
        frame_ms=args.frame_ms,
        timeout=args.timeout,
    )
    started = time.perf_counter()
    gathered = await asyncio.gather(
        *(run_session(session_args) for _ in range(concurrency)),
        return_exceptions=True,
    )
    wall_sec = time.perf_counter() - started
    sessions: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in gathered:
        if isinstance(item, BaseException):
            errors.append(f"{type(item).__name__}: {item}")
        else:
            sessions.append(item)

    raw_final = [float(item["raw_final_visible_sec"]) for item in sessions]
    complete = [float(item["complete_sec"]) for item in sessions]
    revision_delay = [
        float(item["complete_sec"]) - float(item["raw_final_visible_sec"])
        for item in sessions
    ]
    refiner_rpc = [
        float(value)
        for item in sessions
        for value in [item["session_snapshot"]["latency_ms"].get("refiner_rpc_max")]
        if value is not None
    ]
    passed = (
        not errors
        and len(sessions) == concurrency
        and all(item["status"] == "pass" and item["revision_events"] >= 1 for item in sessions)
    )
    return {
        "concurrency": concurrency,
        "status": "pass" if passed else "fail",
        "wall_sec": round(wall_sec, 6),
        "completed_sessions": len(sessions),
        "errors": errors,
        "raw_final_visible_sec": {
            "p50": percentile(raw_final, 0.50),
            "p95": percentile(raw_final, 0.95),
            "max": round(max(raw_final), 6) if raw_final else None,
        },
        "complete_sec": {
            "p50": percentile(complete, 0.50),
            "p95": percentile(complete, 0.95),
            "max": round(max(complete), 6) if complete else None,
        },
        "post_final_revision_sec": {
            "p50": percentile(revision_delay, 0.50),
            "p95": percentile(revision_delay, 0.95),
            "max": round(max(revision_delay), 6) if revision_delay else None,
        },
        "refiner_rpc_max_ms": {
            "p50": percentile(refiner_rpc, 0.50),
            "p95": percentile(refiner_rpc, 0.95),
            "max": round(max(refiner_rpc), 6) if refiner_rpc else None,
        },
        "sessions": sessions,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    levels = []
    for concurrency in args.levels:
        levels.append(await run_level(args, concurrency))
    return {
        "benchmark": "Gate1 + Gate2 paced real-audio concurrency staircase",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(item["status"] == "pass" for item in levels) else "fail",
        "service": args.url,
        "audio": str(args.audio.resolve()),
        "audio_sec_limit": args.audio_sec,
        "frame_ms": args.frame_ms,
        "levels": levels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--audio-sec", type=float, default=5.0)
    parser.add_argument("--frame-ms", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "live_concurrency")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(value <= 0 for value in args.levels):
        raise ValueError("concurrency levels must be positive")
    result = asyncio.run(run(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(args.output_dir / timestamp / "live_concurrency.json", result)
    write_json(args.output_dir / "live_concurrency_latest.json", result)
    summary = {**result, "levels": [{key: value for key, value in level.items() if key != "sessions"} for level in result["levels"]]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
