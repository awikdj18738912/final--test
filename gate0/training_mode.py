#!/usr/bin/env python3
"""Guard a training command so it only runs with both GPUs exclusively free."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()]) if result.returncode == 0 else 0


def active_compute_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else ["nvidia-smi unavailable"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only check exclusive readiness")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Training command after --")
    args = parser.parse_args()
    processes = active_compute_processes()
    ready = gpu_count() >= 2 and not processes
    report = {"training_exclusive_ready": ready, "gpu_count": gpu_count(), "compute_processes": processes}
    print(report)
    if not ready:
        print("Stop realtime/offline services before training.", file=sys.stderr)
        return 2
    if args.check or not args.command:
        return 0
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
