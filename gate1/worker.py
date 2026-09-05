#!/usr/bin/env python3
"""One-GPU Qwen3-ASR worker exposed through a Unix-domain JSON-lines RPC API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def encode_response(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def load_waveform(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    waveform, sample_rate = sf.read(str(audio_path), always_2d=False)
    waveform = np.asarray(waveform)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    if sample_rate != 16000:
        import torch
        import torchaudio.functional as F

        waveform = F.resample(torch.from_numpy(waveform), sample_rate, 16000).numpy().astype(np.float32)
        sample_rate = 16000
    return waveform, sample_rate


class QwenWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Any = None
        self.states: dict[str, Any] = {}
        self.warmup_sec: float | None = None

    def load(self) -> None:
        # Import after CUDA_VISIBLE_DEVICES has already been set by the parent.
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.LLM(
            model=self.args.model,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_model_len,
            max_inference_batch_size=1,
            max_new_tokens=self.args.max_new_tokens,
        )

    def warmup(self) -> None:
        """Run the first ASR request before accepting user traffic."""
        waveform, sample_rate = load_waveform(self.args.warmup_audio)
        started = time.perf_counter()
        if self.args.role == "realtime":
            state = self.model.init_streaming_state(language="Chinese", chunk_size_sec=self.args.chunk_size_sec)
            push_samples = max(1, int(round(0.25 * sample_rate)))
            for offset in range(0, len(waveform), push_samples):
                self.model.streaming_transcribe(waveform[offset : offset + push_samples], state)
            self.model.finish_streaming_transcribe(state)
        else:
            self.model.transcribe(audio=(waveform, sample_rate), language="Chinese")
        self.warmup_sec = round(time.perf_counter() - started, 3)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "health":
            return {
                "role": self.args.role,
                "model": self.args.model,
                "active_sessions": len(self.states),
                "warmup_sec": self.warmup_sec,
            }
        if operation == "start":
            if self.args.role != "realtime":
                raise ValueError("start is only valid for the realtime worker")
            session_id = str(request["session_id"])
            if session_id in self.states:
                raise ValueError("session already exists in realtime worker")
            self.states[session_id] = self.model.init_streaming_state(
                language=str(request.get("language") or "Chinese"),
                chunk_size_sec=float(request.get("chunk_size_sec", self.args.chunk_size_sec)),
            )
            return {"session_id": session_id, "text": ""}
        if operation == "push":
            if self.args.role != "realtime":
                raise ValueError("push is only valid for the realtime worker")
            session_id = str(request["session_id"])
            state = self.states.get(session_id)
            if state is None:
                raise KeyError("unknown realtime session")
            if int(request.get("sample_rate", 16000)) != 16000:
                raise ValueError("realtime worker accepts 16 kHz PCM16 only")
            raw = base64.b64decode(request["pcm16_b64"], validate=True)
            if not raw or len(raw) % 2:
                raise ValueError("PCM16 frame must contain a non-empty even number of bytes")
            waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            self.model.streaming_transcribe(waveform, state)
            return {"session_id": session_id, "text": state.text, "language": state.language}
        if operation == "finish":
            if self.args.role != "realtime":
                raise ValueError("finish is only valid for the realtime worker")
            session_id = str(request["session_id"])
            state = self.states.pop(session_id, None)
            if state is None:
                raise KeyError("unknown realtime session")
            self.model.finish_streaming_transcribe(state)
            return {"session_id": session_id, "text": state.text, "language": state.language}
        if operation == "close":
            session_id = str(request["session_id"])
            self.states.pop(session_id, None)
            return {"session_id": session_id, "closed": True}
        if operation == "offline_transcribe":
            if self.args.role != "offline":
                raise ValueError("offline_transcribe is only valid for the offline worker")
            waveform, sample_rate = load_waveform(Path(str(request["audio_path"])))
            result = self.model.transcribe(
                audio=(waveform, sample_rate), language=str(request.get("language") or "Chinese")
            )[0]
            return {"text": result.text, "language": result.language, "audio_sec": round(len(waveform) / sample_rate, 4)}
        raise ValueError(f"unknown operation: {operation}")


async def serve(worker: QwenWorker, socket_path: Path) -> None:
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
        except Exception as exc:  # RPC error is returned to the API process.
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
    parser.add_argument("--role", choices=["realtime", "offline"], required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--warmup-audio",
        type=Path,
        default=Path(
            "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0913/BAC009S0913W0321.wav"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(serve(QwenWorker(args), args.socket))
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
