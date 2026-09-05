#!/usr/bin/env python3
"""Build a deterministic, duration-stratified AISHELL-1 quality manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/manifests/test_zh-CN.json"
)


def stratum(duration: float) -> str:
    if duration < 5:
        return "short_lt_5s"
    if duration < 10:
        return "medium_5_10s"
    return "long_ge_10s"


def speaker(item: dict[str, Any]) -> str:
    return Path(item["audio_filepath"]).parent.name


def balanced_sample(items: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Round-robin available speakers, after deterministic within-speaker shuffle."""
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_speaker[speaker(item)].append(item)
    for values in by_speaker.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    speakers = sorted(by_speaker)
    cursor = 0
    while len(selected) < count and speakers:
        current = speakers[cursor % len(speakers)]
        if by_speaker[current]:
            selected.append(by_speaker[current].pop())
        if not by_speaker[current]:
            speakers.remove(current)
            if not speakers:
                break
            cursor %= len(speakers)
        else:
            cursor += 1
    if len(selected) != count:
        raise ValueError(f"requested {count} samples but only selected {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "quality_manifest_200.json")
    parser.add_argument("--short", type=int, default=80)
    parser.add_argument("--medium", type=int, default=80)
    parser.add_argument("--long", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if min(args.short, args.medium, args.long) < 0 or args.short + args.medium + args.long <= 0:
        parser.error("stratum counts must be non-negative and total count must be positive")

    source_bytes = args.source.read_bytes()
    population = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    for item in population:
        item["speaker"] = speaker(item)
        item["stratum"] = stratum(float(item["duration"]))
        item["utterance_id"] = Path(item["audio_filepath"]).stem
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in population:
        grouped[item["stratum"]].append(item)

    rng = random.Random(args.seed)
    targets = {
        "short_lt_5s": args.short,
        "medium_5_10s": args.medium,
        "long_ge_10s": args.long,
    }
    selected = [item for name, count in targets.items() for item in balanced_sample(grouped[name], count, rng)]
    rng.shuffle(selected)
    duration_sec = sum(float(item["duration"]) for item in selected)
    manifest = {
        "manifest": "Gate 0.5 AISHELL-1 stratified quality evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": args.seed,
        "selection": {
            "duration_strata": {"short_lt_5s": "duration < 5", "medium_5_10s": "5 <= duration < 10", "long_ge_10s": "duration >= 10"},
            "targets": targets,
            "method": "speaker-balanced round-robin within each stratum, then deterministic global shuffle",
            "purpose": "controlled backend comparison; long samples are intentionally over-sampled, not a population CER estimate",
        },
        "population": {
            "utterances": len(population),
            "speakers": len({item["speaker"] for item in population}),
            "strata": dict(Counter(item["stratum"] for item in population)),
        },
        "summary": {
            "utterances": len(selected),
            "speakers": len({item["speaker"] for item in selected}),
            "audio_sec": round(duration_sec, 4),
            "audio_hours": round(duration_sec / 3600, 4),
            "strata": dict(Counter(item["stratum"] for item in selected)),
            "speaker_counts": dict(sorted(Counter(item["speaker"] for item in selected).items())),
        },
        "samples": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
