#!/usr/bin/env python3
"""GPU Refiner worker exposed through a Unix-domain JSON-lines RPC API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any

from .refiner import TransformersRefiner


def encode_response(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class RefinerWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.refiner: TransformersRefiner | None = None
        self.warmup_sec: float | None = None

    def load(self) -> None:
        self.refiner = TransformersRefiner(self.args.model, max_new_tokens=self.args.max_new_tokens)

    def warmup(self) -> None:
        if self.refiner is None:
            raise RuntimeError("refiner is not loaded")
        started = time.perf_counter()
        self.refiner.rewrite(
            read_only_prefix="",
            active_source_window="这是一条预热文本。",
            trusted_memory=[],
        )
        self.warmup_sec = round(time.perf_counter() - started, 3)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.refiner is None:
            raise RuntimeError("refiner is not loaded")
        operation = request.get("op")
        if operation == "health":
            return {
                "role": "refiner",
                "model": str(self.args.model),
                "load_sec": round(self.refiner.load_sec, 3),
                "warmup_sec": self.warmup_sec,
                "max_new_tokens": self.args.max_new_tokens,
                "stop_token_ids": self.refiner.stop_token_ids,
            }
        if operation == "rewrite":
            return self.refiner.rewrite(
                read_only_prefix=str(request.get("read_only_prefix", "")),
                active_source_window=str(request["active_source_window"]),
                trusted_memory=[str(item) for item in request.get("trusted_memory", [])],
            )
        raise ValueError(f"unknown operation: {operation}")


async def serve(worker: RefinerWorker, socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    worker.load()
    worker.warmup()

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not line:
                return
            request = json.loads(line)
            response = {"ok": True, "result": worker.handle(request)}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        writer.write(encode_response(response))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_connection, path=str(socket_path))
    try:
        async with server:
            await server.serve_forever()
    finally:
        socket_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(serve(RefinerWorker(args), args.socket))
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
