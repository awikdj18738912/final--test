#!/usr/bin/env python3
"""Gate 1 FastAPI service: realtime WebSocket sessions and offline jobs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from gate2.streaming import K1RefinementState
from gate3.rule_router import RouteDecision, route


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(os.environ.get("GATE1_RUNTIME_DIR", ROOT / "gate1" / "runtime"))
REALTIME_MODEL = os.environ.get("GATE1_REALTIME_MODEL", "/home/aim0/data/models/ASR/Qwen3-ASR-1.7B")
OFFLINE_MODEL = os.environ.get("GATE1_OFFLINE_MODEL", "/home/aim0/data/models/ASR/Qwen3-ASR-1.7B")
REALTIME_CHUNK_SEC = float(os.environ.get("GATE1_REALTIME_CHUNK_SEC", "1.0"))
REALTIME_GPU_MEMORY = float(os.environ.get("GATE1_REALTIME_GPU_MEMORY", "0.78"))
OFFLINE_GPU_MEMORY = float(os.environ.get("GATE1_OFFLINE_GPU_MEMORY", "0.50"))
WARMUP_AUDIO = os.environ.get(
    "GATE1_WARMUP_AUDIO", "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0913/BAC009S0913W0321.wav"
)
MAX_FRAME_BYTES = int(os.environ.get("GATE1_MAX_FRAME_BYTES", str(128 * 1024)))
MAX_UPLOAD_BYTES = int(os.environ.get("GATE1_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
GPU1_ROLE = os.environ.get("GATE1_GPU1_ROLE", "offline").lower()
REFINER_MODEL = os.environ.get("GATE1_REFINER_MODEL", "/home/aim0/data/models/ASR/AgenticASR-Refiner")
REFINER_TIMEOUT_SEC = float(os.environ.get("GATE1_REFINER_TIMEOUT_SEC", "10"))
REFINER_MAX_NEW_TOKENS = int(os.environ.get("GATE1_REFINER_MAX_NEW_TOKENS", "256"))
REFINER_ROUTER_MODE = os.environ.get("GATE1_REFINER_ROUTER", "conservative").lower()
COLLECT_LOGPROB_TELEMETRY = os.environ.get("GATE1_COLLECT_LOGPROB_TELEMETRY", "").lower() in {"1", "true", "yes", "on"}


class SessionCreate(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    mode: str = "realtime"
    language: str = "Chinese"
    quality_level: str = "balanced"


@dataclass
class SessionRecord:
    session_id: str
    tenant_id: str
    language: str
    quality_level: str
    created_at: float
    raw_text: str = ""
    text: str = ""
    result_version: int = 0
    input_seq_no: int = 0
    active: bool = False
    final: bool = False
    stream_started_at: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    asr_stream_metrics: dict[str, Any] = field(default_factory=dict)
    refinement: K1RefinementState | None = field(default=None, repr=False)
    refinement_tasks: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    refinement_results: list[dict[str, Any]] = field(default_factory=list)
    refinement_disabled_reason: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass
class OfflineJob:
    job_id: str
    tenant_id: str
    language: str
    source_name: str
    source_path: Path
    status: str
    created_at: float
    updated_at: float
    text: str | None = None
    audio_sec: float | None = None
    error: str | None = None
    cancel_requested: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class WorkerProcess:
    def __init__(self, role: str, socket_path: Path, model: str, gpu: int, gpu_memory_utilization: float):
        self.role = role
        self.socket_path = socket_path
        self.model = model
        self.gpu = gpu
        self.gpu_memory_utilization = gpu_memory_utilization
        self.process: subprocess.Popen[str] | None = None
        self.log_path = RUNTIME_DIR / f"{role}.worker.log"

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "gate1.worker",
            "--role",
            self.role,
            "--socket",
            str(self.socket_path),
            "--model",
            self.model,
            "--warmup-audio",
            WARMUP_AUDIO,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            "8192",
            "--max-new-tokens",
            "256",
        ]
        if self.role == "realtime":
            command.extend(["--chunk-size-sec", str(REALTIME_CHUNK_SEC)])
            if COLLECT_LOGPROB_TELEMETRY:
                command.append("--collect-logprob-telemetry")
        return command

    async def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.socket_path.unlink(missing_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}:{existing_pythonpath}"
        with self.log_path.open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                self.command(),
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                # vLLM starts EngineCore as a child.  Isolating this worker in
                # its own process group lets shutdown release GPU memory even
                # if that child does not exit with the Python wrapper.
                start_new_session=True,
            )
        deadline = time.monotonic() + 180
        last_error = "worker socket was not created"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.role} worker exited; inspect {self.log_path}")
            try:
                await self.rpc({"op": "health"}, timeout=2)
                return
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(0.5)
        raise RuntimeError(f"{self.role} worker did not become ready: {last_error}")

    async def rpc(self, payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"{self.role} worker is not running")
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(self.socket_path)), timeout=timeout)
        try:
            writer.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            await asyncio.wait_for(writer.drain(), timeout=timeout)
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise RuntimeError(f"{self.role} worker closed RPC connection")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "unknown worker error"))
            return response["result"]
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=10)
        self.socket_path.unlink(missing_ok=True)


class RefinerProcess(WorkerProcess):
    def __init__(self, socket_path: Path, model: str, gpu: int) -> None:
        super().__init__("refiner", socket_path, model, gpu, 0.0)

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "gate2.worker",
            "--socket",
            str(self.socket_path),
            "--model",
            self.model,
            "--max-new-tokens",
            str(REFINER_MAX_NEW_TOKENS),
        ]


class ServiceState:
    def __init__(self) -> None:
        if GPU1_ROLE not in {"offline", "refiner"}:
            raise ValueError("GATE1_GPU1_ROLE must be offline or refiner")
        if REFINER_ROUTER_MODE not in {"conservative", "all", "off"}:
            raise ValueError("GATE1_REFINER_ROUTER must be conservative, all, or off")
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.realtime = WorkerProcess("realtime", RUNTIME_DIR / "realtime.sock", REALTIME_MODEL, 0, REALTIME_GPU_MEMORY)
        self.offline = (
            WorkerProcess("offline", RUNTIME_DIR / "offline.sock", OFFLINE_MODEL, 1, OFFLINE_GPU_MEMORY)
            if GPU1_ROLE == "offline"
            else None
        )
        self.refiner = (
            RefinerProcess(RUNTIME_DIR / "refiner.sock", REFINER_MODEL, 1)
            if GPU1_ROLE == "refiner"
            else None
        )
        self.sessions: dict[str, SessionRecord] = {}
        self.jobs: dict[str, OfflineJob] = {}
        self.idempotency: dict[tuple[str, str, str], str] = {}

    async def start(self) -> None:
        await self.realtime.start()
        if self.offline is not None:
            await self.offline.start()
        if self.refiner is not None:
            await self.refiner.start()

    def stop(self) -> None:
        self.realtime.stop()
        if self.offline is not None:
            self.offline.stop()
        if self.refiner is not None:
            self.refiner.stop()

    @staticmethod
    def public_session(record: SessionRecord) -> dict[str, Any]:
        return {
            "session_id": record.session_id,
            "tenant_id": record.tenant_id,
            "language": record.language,
            "quality_level": record.quality_level,
            "created_at": record.created_at,
            "raw_text": record.raw_text,
            "text": record.text,
            "result_version": record.result_version,
            "input_seq_no": record.input_seq_no,
            "active": record.active,
            "final": record.final,
            "latency_ms": record.latency_ms,
            "asr_stream_metrics": record.asr_stream_metrics,
            "refinement_enabled": record.refinement is not None,
            "refiner_router_mode": REFINER_ROUTER_MODE if record.refinement is not None else None,
            "refinement_disabled_reason": record.refinement_disabled_reason,
            "refinement_results": record.refinement_results,
        }

    @staticmethod
    def public_job(job: OfflineJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "language": job.language,
            "source_name": job.source_name,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "text": job.text,
            "audio_sec": job.audio_sec,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
            "latency_ms": job.latency_ms,
        }


def session_or_404(state: ServiceState, session_id: str, tenant_id: str) -> SessionRecord:
    session = state.sessions.get(session_id)
    if session is None or session.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def job_or_404(state: ServiceState, job_id: str, tenant_id: str) -> OfflineJob:
    job = state.jobs.get(job_id)
    if job is None or job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="offline job not found")
    return job


def transcript_event(session: SessionRecord, event: str, text: str, is_final: bool) -> dict[str, Any]:
    prior = session.text
    session.result_version += 1
    session.text = text
    session.final = is_final
    return {
        "event": event,
        "session_id": session.session_id,
        "window_id": "full_transcript",
        "base_version": session.result_version - 1,
        "result_version": session.result_version,
        "base_hash": hashlib.sha256(prior.encode("utf-8")).hexdigest(),
        "is_final": is_final,
        "text": text,
    }


def add_latency(session: SessionRecord, name: str, value_ms: float) -> None:
    total_key = f"{name}_total"
    max_key = f"{name}_max"
    count_key = f"{name}_count"
    session.latency_ms[total_key] = round(session.latency_ms.get(total_key, 0.0) + value_ms, 3)
    session.latency_ms[max_key] = round(max(session.latency_ms.get(max_key, 0.0), value_ms), 3)
    session.latency_ms[count_key] = session.latency_ms.get(count_key, 0.0) + 1


async def websocket_sender(websocket: WebSocket, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
    while True:
        event = await queue.get()
        try:
            if event is None:
                return
            await websocket.send_json(event)
        finally:
            queue.task_done()


def update_refinement_hypothesis(
    session: SessionRecord, hypothesis: str, *, is_final: bool
) -> tuple[str, list[Any]]:
    session.raw_text = hypothesis
    if session.refinement is None:
        return hypothesis, []
    try:
        closed = session.refinement.update_hypothesis(hypothesis, is_final=is_final)
        return session.refinement.render(), closed
    except ValueError as exc:
        session.refinement_disabled_reason = str(exc)
        session.refinement = None
        return hypothesis, []


async def refine_span(
    state: ServiceState,
    session: SessionRecord,
    span_id: str,
    queue: asyncio.Queue[dict[str, Any] | None],
    router_decision: RouteDecision | None = None,
) -> None:
    if state.refiner is None:
        return
    queued_at = time.perf_counter()
    async with session.lock:
        refinement = session.refinement
        if refinement is None or span_id not in refinement.sources:
            return
        window = refinement.window(span_id)
    rpc_started = time.perf_counter()
    try:
        result = await state.refiner.rpc(
            {
                "op": "rewrite",
                "read_only_prefix": window.read_only_prefix,
                "active_source_window": window.source_text,
                "trusted_memory": [],
            },
            timeout=REFINER_TIMEOUT_SEC,
        )
        rpc_ms = (time.perf_counter() - rpc_started) * 1000
        inference_ms = float(result["inference_sec"]) * 1000
        queue_ms = max(0.0, rpc_ms - inference_ms)
        generation_complete = (
            result["finish_reason"] != "length"
            or bool(result["key_suffix_present"])
            or str(result["text"]) == window.source_text
        )
        async with session.lock:
            refinement = session.refinement
            if refinement is None or span_id not in refinement.sources:
                return
            decision = refinement.apply(
                span_id=span_id,
                clean_text=str(result["text"]),
                tenant_id=session.tenant_id,
                generation_complete=generation_complete,
            )
            add_latency(session, "refiner_rpc", rpc_ms)
            add_latency(session, "refiner_inference", inference_ms)
            add_latency(session, "refiner_queue", queue_ms)
            add_latency(session, "revision", (time.perf_counter() - queued_at) * 1000)
            rendered = refinement.render()
            if decision.event == "replace" and rendered != session.text:
                event = transcript_event(session, "revision", rendered, session.final)
            else:
                current_hash = hashlib.sha256(session.text.encode("utf-8")).hexdigest()
                event = {
                    "event": f"refiner_{decision.event}",
                    "session_id": session.session_id,
                    "base_version": session.result_version,
                    "result_version": session.result_version,
                    "base_hash": current_hash,
                    "is_final": session.final,
                    "text": session.text,
                }
            event.update(
                {
                    "source_window_id": decision.window.window_id,
                    "source_span_id": span_id,
                    "source_text": decision.window.source_text,
                    "clean_text": decision.clean_text,
                    "patches": [asdict(patch) for patch in decision.patches],
                    "validation": asdict(decision.validation),
                    "model_finish_reason": result["finish_reason"],
                    "model_stop_token_id": result["stop_token_id"],
                    "model_generated_tokens": result["generated_tokens"],
                    "model_keys": result["keys"],
                    "refiner_inference_ms": round(inference_ms, 3),
                    "refiner_rpc_ms": round(rpc_ms, 3),
                    "router": router_decision.to_dict() if router_decision is not None else None,
                }
            )
            session.refinement_results.append(
                {
                    "event": event["event"],
                    "source_window_id": decision.window.window_id,
                    "source_span_id": span_id,
                    "validation": asdict(decision.validation),
                    "model_finish_reason": result["finish_reason"],
                    "model_stop_token_id": result["stop_token_id"],
                    "model_generated_tokens": result["generated_tokens"],
                    "refiner_inference_ms": round(inference_ms, 3),
                    "refiner_rpc_ms": round(rpc_ms, 3),
                    "router": router_decision.to_dict() if router_decision is not None else None,
                }
            )
        await queue.put(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        async with session.lock:
            add_latency(session, "refiner_error", (time.perf_counter() - rpc_started) * 1000)
            error_event = {
                "event": "refiner_error",
                "session_id": session.session_id,
                "source_span_id": span_id,
                "base_version": session.result_version,
                "result_version": session.result_version,
                "base_hash": hashlib.sha256(session.text.encode("utf-8")).hexdigest(),
                "is_final": session.final,
                "text": session.text,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            session.refinement_results.append(error_event.copy())
        await queue.put(error_event)


def schedule_refinements(
    state: ServiceState,
    session: SessionRecord,
    spans: list[Any],
    queue: asyncio.Queue[dict[str, Any] | None],
) -> None:
    if state.refiner is None:
        return
    for span in spans:
        decision = route_span(span.text)
        if not decision.call_refiner:
            skipped = {
                "event": "refiner_skipped",
                "session_id": session.session_id,
                "source_span_id": span.span_id,
                "source_text": span.text,
                "base_version": session.result_version,
                "result_version": session.result_version,
                "base_hash": hashlib.sha256(session.text.encode("utf-8")).hexdigest(),
                "is_final": session.final,
                "text": session.text,
                "router": decision.to_dict(),
            }
            session.refinement_results.append(skipped.copy())
            queue.put_nowait(skipped)
            continue
        task = asyncio.create_task(
            refine_span(state, session, span.span_id, queue, decision),
            name=f"refine-{session.session_id}-{span.span_id}",
        )
        session.refinement_tasks.add(task)
        task.add_done_callback(session.refinement_tasks.discard)


def route_span(raw_text: str) -> RouteDecision:
    """Choose a Refiner policy while retaining explicit benchmark escape hatches."""

    if REFINER_ROUTER_MODE == "all":
        return RouteDecision(True, 0, ("router_all",))
    if REFINER_ROUTER_MODE == "off":
        return RouteDecision(False, 0, ("router_off",))
    return route(raw_text)


async def wait_for_refinements(session: SessionRecord) -> None:
    while session.refinement_tasks:
        await asyncio.gather(*list(session.refinement_tasks), return_exceptions=True)


async def start_offline_job(state: ServiceState, job: OfflineJob) -> None:
    if state.offline is None:
        raise RuntimeError("offline worker is disabled")
    if job.cancel_requested:
        job.status = "cancelled"
        job.updated_at = time.time()
        job.source_path.unlink(missing_ok=True)
        return
    job.status = "running"
    job.updated_at = time.time()
    job.latency_ms["queue_wait"] = round((job.updated_at - job.created_at) * 1000, 3)
    inference_started = time.perf_counter()
    try:
        result = await state.offline.rpc(
            {"op": "offline_transcribe", "audio_path": str(job.source_path), "language": job.language}, timeout=1800
        )
        if job.cancel_requested:
            job.status = "cancelled"
        else:
            job.status = "succeeded"
            job.text = result["text"]
            job.audio_sec = result["audio_sec"]
    except Exception as exc:
        job.status = "cancelled" if job.cancel_requested else "failed"
        if not job.cancel_requested:
            job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.updated_at = time.time()
        job.latency_ms["worker_rpc"] = round((time.perf_counter() - inference_started) * 1000, 3)
        job.latency_ms["end_to_end"] = round((job.updated_at - job.created_at) * 1000, 3)
        job.source_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ServiceState()
    app.state.service = state
    try:
        await state.start()
    except Exception:
        state.stop()
        raise
    try:
        yield
    finally:
        for session in state.sessions.values():
            for task in session.refinement_tasks:
                if not task.done():
                    task.cancel()
        for job in state.jobs.values():
            if job.task is not None and not job.task.done():
                job.task.cancel()
        state.stop()


app = FastAPI(title="Dual-3090 ASR Gate 1", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    state: ServiceState = app.state.service
    result = {
        "status": "ok",
        "realtime": await state.realtime.rpc({"op": "health"}),
        "gpu1_role": GPU1_ROLE,
        "refiner_router_mode": REFINER_ROUTER_MODE if GPU1_ROLE == "refiner" else None,
        "sessions": len(state.sessions),
        "jobs": len(state.jobs),
    }
    result["offline"] = await state.offline.rpc({"op": "health"}) if state.offline is not None else None
    result["refiner"] = await state.refiner.rpc({"op": "health"}) if state.refiner is not None else None
    return result


@app.post("/sessions", status_code=201)
async def create_session(request: SessionCreate, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    if request.mode != "realtime":
        raise HTTPException(status_code=422, detail="Gate 1 currently supports mode=realtime only")
    state: ServiceState = app.state.service
    if idempotency_key:
        existing = state.idempotency.get((request.tenant_id, "session", idempotency_key))
        if existing:
            return ServiceState.public_session(state.sessions[existing])
    session = SessionRecord(
        session_id=f"ses_{uuid.uuid4().hex}",
        tenant_id=request.tenant_id,
        language=request.language,
        quality_level=request.quality_level,
        created_at=time.time(),
    )
    if state.refiner is not None:
        session.refinement = K1RefinementState(session.session_id)
    state.sessions[session.session_id] = session
    if idempotency_key:
        state.idempotency[(request.tenant_id, "session", idempotency_key)] = session.session_id
    return ServiceState.public_session(session)


@app.get("/sessions/{session_id}/result")
async def get_session_result(session_id: str, tenant_id: str = Query(min_length=1)) -> dict[str, Any]:
    state: ServiceState = app.state.service
    return ServiceState.public_session(session_or_404(state, session_id, tenant_id))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, tenant_id: str = Query(min_length=1)) -> dict[str, Any]:
    state: ServiceState = app.state.service
    session = session_or_404(state, session_id, tenant_id)
    async with session.lock:
        if session.active:
            await state.realtime.rpc({"op": "close", "session_id": session.session_id})
            session.active = False
        session.final = True
        for task in list(session.refinement_tasks):
            if not task.done():
                task.cancel()
    state.sessions.pop(session_id, None)
    stale_idempotency_keys = [
        key for key, value in state.idempotency.items()
        if key[0] == tenant_id and key[1] == "session" and value == session_id
    ]
    for key in stale_idempotency_keys:
        state.idempotency.pop(key, None)
    return {"session_id": session_id, "deleted": True}


@app.websocket("/sessions/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str, tenant_id: str = Query(min_length=1)) -> None:
    state: ServiceState = app.state.service
    session = state.sessions.get(session_id)
    if session is None or session.tenant_id != tenant_id:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    send_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sender_task = asyncio.create_task(websocket_sender(websocket, send_queue), name=f"send-{session_id}")
    normal_completion = False
    try:
        async with session.lock:
            if session.active or session.final:
                await websocket.close(code=4409)
                return
            session.stream_started_at = time.perf_counter()
            state_started = time.perf_counter()
            await state.realtime.rpc(
                {
                    "op": "start",
                    "session_id": session.session_id,
                    "language": session.language,
                    "chunk_size_sec": REALTIME_CHUNK_SEC,
                }
            )
            session.latency_ms["worker_start"] = round((time.perf_counter() - state_started) * 1000, 3)
            session.latency_ms["worker_rpc_total"] = 0.0
            session.latency_ms["worker_rpc_max"] = 0.0
            session.active = True
        await send_queue.put(
            {
                "event": "ready",
                "session_id": session.session_id,
                "sample_rate": 16000,
                "format": "pcm_s16le_mono",
                "chunk_size_sec": REALTIME_CHUNK_SEC,
                "refinement_enabled": session.refinement is not None,
                "refiner_window_k": 1 if session.refinement is not None else None,
                "refiner_router_mode": REFINER_ROUTER_MODE if session.refinement is not None else None,
            }
        )
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            audio: bytes | None = message.get("bytes")
            explicit_sequence: int | None = None
            if audio is None and message.get("text") is not None:
                payload = json.loads(message["text"])
                event = payload.get("event")
                if event == "end":
                    async with session.lock:
                        inference_started = time.perf_counter()
                        result = await state.realtime.rpc({"op": "finish", "session_id": session.session_id})
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        session.latency_ms["worker_rpc_total"] = round(session.latency_ms["worker_rpc_total"] + inference_ms, 3)
                        session.latency_ms["worker_rpc_max"] = round(max(session.latency_ms["worker_rpc_max"], inference_ms), 3)
                        if "first_result" not in session.latency_ms and result["text"] and session.stream_started_at is not None:
                            session.latency_ms["first_result"] = round((time.perf_counter() - session.stream_started_at) * 1000, 3)
                        if session.stream_started_at is not None:
                            session.latency_ms["final_result"] = round((time.perf_counter() - session.stream_started_at) * 1000, 3)
                        session.active = False
                        session.asr_stream_metrics = dict(result.get("stream_metrics") or session.asr_stream_metrics)
                        rendered, closed = update_refinement_hypothesis(session, result["text"], is_final=True)
                        final_event = transcript_event(session, "final", rendered, True)
                        final_event["raw_text"] = result["text"]
                        final_event["asr_stream_metrics"] = session.asr_stream_metrics
                        send_queue.put_nowait(final_event)
                    schedule_refinements(state, session, closed, send_queue)
                    await wait_for_refinements(session)
                    await asyncio.wait_for(send_queue.join(), timeout=max(30.0, REFINER_TIMEOUT_SEC * 2))
                    await websocket.send_json(
                        {
                            "event": "complete",
                            "session_id": session.session_id,
                            "result_version": session.result_version,
                            "is_final": True,
                            "raw_text": session.raw_text,
                            "text": session.text,
                            "refinement_disabled_reason": session.refinement_disabled_reason,
                            "refinement_results": len(session.refinement_results),
                        }
                    )
                    normal_completion = True
                    return
                if event != "audio":
                    raise ValueError("text frames must be an audio event or an end event")
                explicit_sequence = int(payload["sequence_id"])
                audio = base64.b64decode(payload["pcm16_b64"], validate=True)
                if int(payload.get("sample_rate", 16000)) != 16000:
                    raise ValueError("sample_rate must be 16000")
            if audio is None:
                continue
            if len(audio) == 0 or len(audio) % 2 or len(audio) > MAX_FRAME_BYTES:
                raise ValueError("audio frame must be non-empty PCM16 and within the configured frame limit")
            async with session.lock:
                expected_sequence = session.input_seq_no + 1
                if explicit_sequence is not None and explicit_sequence != expected_sequence:
                    raise ValueError(f"sequence_id must be {expected_sequence}")
                session.input_seq_no = expected_sequence
                inference_started = time.perf_counter()
                result = await state.realtime.rpc(
                    {
                        "op": "push",
                        "session_id": session.session_id,
                        "sample_rate": 16000,
                        "pcm16_b64": base64.b64encode(audio).decode("ascii"),
                    }
                )
                inference_ms = (time.perf_counter() - inference_started) * 1000
                session.latency_ms["worker_rpc_total"] = round(session.latency_ms["worker_rpc_total"] + inference_ms, 3)
                session.latency_ms["worker_rpc_max"] = round(max(session.latency_ms["worker_rpc_max"], inference_ms), 3)
                if result["text"] == session.raw_text:
                    session.asr_stream_metrics = dict(result.get("stream_metrics") or session.asr_stream_metrics)
                    continue
                if "first_result" not in session.latency_ms and session.stream_started_at is not None:
                    session.latency_ms["first_result"] = round((time.perf_counter() - session.stream_started_at) * 1000, 3)
                rendered, closed = update_refinement_hypothesis(session, result["text"], is_final=False)
                session.asr_stream_metrics = dict(result.get("stream_metrics") or session.asr_stream_metrics)
                if rendered != session.text:
                    partial_event = transcript_event(session, "partial", rendered, False)
                    partial_event["raw_text"] = result["text"]
                    partial_event["asr_stream_metrics"] = session.asr_stream_metrics
                    send_queue.put_nowait(partial_event)
            schedule_refinements(state, session, closed, send_queue)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with suppress(Exception):
            await send_queue.put({"event": "error", "session_id": session_id, "detail": str(exc)})
            await asyncio.wait_for(send_queue.join(), timeout=2)
    finally:
        if not normal_completion:
            for task in list(session.refinement_tasks):
                if not task.done():
                    task.cancel()
        async with session.lock:
            if session.active:
                with suppress(Exception):
                    await state.realtime.rpc({"op": "close", "session_id": session.session_id})
                session.active = False
        sender_task.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await sender_task


@app.post("/offline/jobs", status_code=202)
async def create_offline_job(
    file: UploadFile = File(...),
    tenant_id: str = Form(..., min_length=1),
    language: str = Form("Chinese"),
    idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    state: ServiceState = app.state.service
    if state.offline is None:
        raise HTTPException(status_code=503, detail="offline ASR is disabled while GPU1 runs the Refiner")
    if idempotency_key:
        existing = state.idempotency.get((tenant_id, "offline", idempotency_key))
        if existing:
            return ServiceState.public_job(state.jobs[existing])
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio file exceeds the configured upload limit")
    job_id = f"job_{uuid.uuid4().hex}"
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    uploads_dir = RUNTIME_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    source_path = uploads_dir / f"{job_id}{suffix}"
    source_path.write_bytes(content)
    now = time.time()
    job = OfflineJob(
        job_id=job_id,
        tenant_id=tenant_id,
        language=language,
        source_name=file.filename or source_path.name,
        source_path=source_path,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    state.jobs[job_id] = job
    if idempotency_key:
        state.idempotency[(tenant_id, "offline", idempotency_key)] = job_id
    job.task = asyncio.create_task(start_offline_job(state, job), name=job_id)
    return ServiceState.public_job(job)


@app.get("/offline/jobs/{job_id}")
async def get_offline_job(job_id: str, tenant_id: str = Query(min_length=1)) -> dict[str, Any]:
    state: ServiceState = app.state.service
    return ServiceState.public_job(job_or_404(state, job_id, tenant_id))


@app.post("/offline/jobs/{job_id}/cancel")
async def cancel_offline_job(job_id: str, tenant_id: str = Query(min_length=1)) -> dict[str, Any]:
    state: ServiceState = app.state.service
    job = job_or_404(state, job_id, tenant_id)
    if job.status in {"succeeded", "failed", "cancelled"}:
        return ServiceState.public_job(job)
    job.cancel_requested = True
    job.updated_at = time.time()
    if job.status == "queued":
        job.status = "cancelled"
        job.source_path.unlink(missing_ok=True)
    return ServiceState.public_job(job)
