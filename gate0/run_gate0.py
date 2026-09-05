#!/usr/bin/env python3
"""Reproducible Gate 0 checks for the fixed dual-RTX-3090 topology.

Each model stage runs in its own subprocess. This is intentional: vLLM uses
spawned worker processes and should not share a Python process with the
Transformers Refiner benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = Path(
    "/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0913/BAC009S0913W0321.wav"
)
DEFAULT_ASR = Path("/home/aim0/data/models/ASR/Qwen3-ASR-1.7B")
DEFAULT_REFINER = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def nvidia_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.used,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(query)
    gpus: list[dict[str, Any]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            try:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "driver": fields[2],
                        "memory_used_mib": int(fields[3]),
                        "memory_total_mib": int(fields[4]),
                        "compute_capability": fields[5],
                    }
                )
            except ValueError:
                continue
    return {"ok": result.returncode == 0, "gpus": gpus, "stderr": result.stderr.strip()}


def compute_processes() -> list[dict[str, str]]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, str]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                processes.append({"gpu_uuid": fields[0], "pid": fields[1], "process": fields[2], "memory_mib": fields[3]})
    return processes


class MemorySampler:
    def __init__(self, gpu_index: int, interval: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        snap = nvidia_snapshot()
        gpu = next((item for item in snap["gpus"] if item["index"] == self.gpu_index), None)
        if gpu is not None:
            self.samples.append({"time": time.time(), **gpu})

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name=f"gpu{self.gpu_index}-memory", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"peak_used_mib": None, "peak_used_gib": None, "samples": 0}
        peak = max(self.samples, key=lambda item: item["memory_used_mib"])
        total = peak["memory_total_mib"]
        return {
            "peak_used_mib": peak["memory_used_mib"],
            "peak_used_gib": round(peak["memory_used_mib"] / 1024, 3),
            "peak_free_gib": round((total - peak["memory_used_mib"]) / 1024, 3),
            "samples": len(self.samples),
        }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_result() -> dict[str, Any]:
    packages = {}
    try:
        from importlib import metadata

        for package in ["qwen-asr", "vllm", "torch", "torchaudio", "transformers", "soundfile", "numpy", "fastapi", "uvicorn", "websockets"]:
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None
    except Exception as exc:  # pragma: no cover - only used on broken Python installs
        packages["error"] = repr(exc)

    torch_info: dict[str, Any] = {"imported": False}
    try:
        import torch

        torch_info = {
            "imported": True,
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_info["error"] = repr(exc)

    import_checks = {}
    for module_name in ["qwen_asr", "vllm"]:
        try:
            __import__(module_name)
            import_checks[module_name] = True
        except Exception as exc:
            import_checks[module_name] = f"{type(exc).__name__}: {exc}"

    nvidia = nvidia_snapshot()
    required = ["qwen-asr", "vllm", "torch", "torchaudio", "transformers", "soundfile", "numpy"]
    paths = {
        "qwen_asr_17b": str(DEFAULT_ASR),
        "qwen_asr_06b": "/home/aim0/data/models/ASR/Qwen3-ASR-0.6B",
        "refiner": str(DEFAULT_REFINER),
        "audio": str(DEFAULT_AUDIO),
    }
    path_check = {name: Path(value).exists() for name, value in paths.items()}
    passed = (
        nvidia["ok"]
        and len(nvidia["gpus"]) >= 2
        and torch_info.get("cuda_available") is True
        and torch_info.get("device_count", 0) >= 2
        and all(packages.get(item) for item in required)
        and all(value is True for value in import_checks.values())
        and all(path_check.values())
    )
    return {
        "stage": "preflight",
        "timestamp": now_iso(),
        "status": "pass" if passed else "fail",
        "platform": {"system": platform.system(), "release": platform.release(), "python": sys.version},
        "packages": packages,
        "imports": import_checks,
        "torch": torch_info,
        "nvidia": nvidia,
        "paths": path_check,
        "compute_processes": compute_processes(),
    }


def training_check_result() -> dict[str, Any]:
    nvidia = nvidia_snapshot()
    processes = compute_processes()
    passed = nvidia["ok"] and len(nvidia["gpus"]) >= 2 and not processes
    return {
        "stage": "training_check",
        "timestamp": now_iso(),
        "status": "pass" if passed else "fail",
        "training_exclusive_ready": passed,
        "message": "Both GPUs are visible and no CUDA compute process is active." if passed else "Stop all GPU services before starting training.",
        "nvidia": nvidia,
        "compute_processes": processes,
    }


def load_waveform(audio_path: Path) -> tuple[Any, int]:
    import numpy as np
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


def qwen_stage(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from qwen_asr import Qwen3ASRModel

    audio_path = Path(args.audio)
    waveform, sample_rate = load_waveform(audio_path)
    sampler = MemorySampler(args.gpu)
    sampler.start()
    load_start = time.perf_counter()
    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    load_sec = time.perf_counter() - load_start

    if args.stage == "offline":
        infer_start = time.perf_counter()
        result = model.transcribe(audio=(waveform, sample_rate), language=args.language)[0]
        infer_sec = time.perf_counter() - infer_start
        text = result.text
        language = result.language
        ttfp_sec = infer_sec
    else:
        state = model.init_streaming_state(language=args.language, chunk_size_sec=args.chunk_size_sec)
        stream_start = time.perf_counter()
        ttfp_sec = None
        push_samples = max(1, int(round(args.input_push_sec * sample_rate)))
        for offset in range(0, len(waveform), push_samples):
            model.streaming_transcribe(waveform[offset : offset + push_samples], state)
            if ttfp_sec is None and state.text:
                ttfp_sec = time.perf_counter() - stream_start
        model.finish_streaming_transcribe(state)
        infer_sec = time.perf_counter() - stream_start
        text = state.text
        language = state.language

    sampler.stop()
    audio_sec = len(waveform) / sample_rate
    rtf = infer_sec / audio_sec if audio_sec else None
    result = {
        "stage": args.stage,
        "timestamp": now_iso(),
        "status": "pass" if text else "fail",
        "gpu": args.gpu,
        "model": args.model,
        "audio": str(audio_path),
        "audio_sec": round(audio_sec, 3),
        "language": language,
        "text": text,
        "load_sec": round(load_sec, 3),
        "inference_sec": round(infer_sec, 3),
        "ttfp_sec": round(ttfp_sec, 3) if ttfp_sec is not None else None,
        "rtf": round(rtf, 4) if rtf is not None else None,
        "memory": sampler.summary(),
        "config": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "chunk_size_sec": args.chunk_size_sec if args.stage == "realtime" else None,
            "input_push_sec": args.input_push_sec if args.stage == "realtime" else None,
        },
    }
    del model
    return result


def refiner_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    model_path = Path(args.refiner_model)
    sampler = MemorySampler(args.gpu)
    sampler.start()
    load_start = time.perf_counter()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
    )
    tokenizer.chat_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0", use_safetensors=True)
    load_sec = time.perf_counter() - load_start
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "请将以下口语转为简洁的书面语，只输出改写后的文本：我明天，不是后天去开会。",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    infer_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.refiner_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    infer_sec = time.perf_counter() - infer_start
    raw_generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    # The checked-in CPM-v2 checkpoint sometimes echoes the instruction before
    # its answer. Keep the raw value for audit, but expose the clean suffix used
    # by the Refiner contract.
    generated = raw_generated
    if "：" in generated:
        generated = generated.rsplit("：", 1)[-1].strip()
    elif ":" in generated:
        generated = generated.rsplit(":", 1)[-1].strip()
    sampler.stop()
    result = {
        "stage": "refiner",
        "timestamp": now_iso(),
        "status": "pass" if generated else "fail",
        "gpu": args.gpu,
        "model": str(model_path),
        "execution": "serialized_arbiter",
        "input": "我明天，不是后天去开会。",
        "raw_text": raw_generated,
        "text": generated,
        "load_sec": round(load_sec, 3),
        "inference_sec": round(infer_sec, 3),
        "ttfp_sec": round(infer_sec, 3),
        "rtf": None,
        "memory": sampler.summary(),
        "config": {"dtype": "bfloat16", "max_new_tokens": args.refiner_max_new_tokens},
    }
    del model
    return result


def stage_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "preflight":
        return environment_result()
    if args.stage == "training-check":
        return training_check_result()
    if args.stage in {"realtime", "offline"}:
        return qwen_stage(args)
    if args.stage == "refiner":
        return refiner_stage(args)
    raise ValueError(f"unknown stage: {args.stage}")


def child_command(args: argparse.Namespace, stage: str, output: Path, gpu: int | None) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--output", str(output)]
    if gpu is not None:
        command += ["--gpu", str(gpu)]
    if stage in {"realtime", "offline"}:
        command += [
            "--model", args.realtime_model if stage == "realtime" else args.offline_model,
            "--audio", args.audio,
            "--language", args.language,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization if stage == "realtime" else args.offline_gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--chunk-size-sec", str(args.chunk_size_sec),
            "--input-push-sec", str(args.input_push_sec),
        ]
    if stage == "refiner":
        command += ["--refiner-model", args.refiner_model, "--refiner-max-new-tokens", str(args.refiner_max_new_tokens)]
    return command


def run_child(args: argparse.Namespace, stage: str, gpu: int | None, output_dir: Path) -> dict[str, Any]:
    output = output_dir / f"{stage.replace('-', '_')}.json"
    log = output_dir / f"{stage.replace('-', '_')}.log"
    command = child_command(args, stage, output, gpu)
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        result = {
            "stage": stage,
            "timestamp": now_iso(),
            "status": "fail",
            "error": f"stage exited {completed.returncode}; inspect {log}",
        }
        write_json(output, result)
    result["exit_code"] = completed.returncode
    return result


def all_stages(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    # GPU1 stages are deliberately serialized: this is the MVP GPU1 Arbiter.
    results = {
        "preflight": run_child(args, "preflight", None, output_dir),
        "realtime": run_child(args, "realtime", 0, output_dir),
        "offline": run_child(args, "offline", 1, output_dir),
        "refiner": run_child(args, "refiner", 1, output_dir),
        "training_check": run_child(args, "training-check", None, output_dir),
    }
    statuses = {name: value.get("status") == "pass" for name, value in results.items()}
    realtime = results["realtime"]
    acceptance = {
        "realtime_ttfp_target_sec": 1.5,
        "realtime_rtf_target": 1.0,
        "realtime_ttfp_target_met": realtime.get("ttfp_sec") is not None and realtime["ttfp_sec"] <= 1.5,
        "realtime_rtf_target_met": realtime.get("rtf") is not None and realtime["rtf"] < 1.0,
        "realtime_capacity_boundary": not (
            realtime.get("ttfp_sec") is not None
            and realtime.get("rtf") is not None
            and realtime["ttfp_sec"] <= 1.5
            and realtime["rtf"] < 1.0
        ),
        "refiner_is_online_tier": False,
        "refiner_note": "Local AgenticASR-Refiner is approximately 4B and measured only as an offline/heavy baseline.",
    }
    functional_pass = all(statuses.values())
    performance_pass = acceptance["realtime_ttfp_target_met"] and acceptance["realtime_rtf_target_met"]
    aggregate = {
        "gate": "Gate 0",
        "timestamp": now_iso(),
        "status": "pass" if functional_pass and performance_pass else ("pass_with_capacity_boundary" if functional_pass else "fail"),
        "topology": "GPU0 realtime ASR; GPU1 serialized offline ASR and Refiner; training exclusive",
        "serialized_arbiter": True,
        "stages": results,
        "stage_pass": statuses,
        "acceptance": acceptance,
        "functional_pass": functional_pass,
        "performance_pass": performance_pass,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "gate0.json", aggregate)
    write_json(Path(args.output_dir) / "latest.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "preflight", "realtime", "offline", "refiner", "training-check"], default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default=str(DEFAULT_ASR))
    parser.add_argument("--realtime-model", default=str(DEFAULT_ASR))
    parser.add_argument("--offline-model", default=str(DEFAULT_ASR))
    parser.add_argument("--refiner-model", default=str(DEFAULT_REFINER))
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--offline-gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--input-push-sec", type=float, default=0.5)
    parser.add_argument("--refiner-max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"realtime", "offline", "refiner"}:
        # Map the requested physical GPU to CUDA device 0 inside this child.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.stage == "all":
        result = all_stages(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"pass", "pass_with_capacity_boundary"} else 1

    result = stage_main(args)
    output = args.output or (args.output_dir / f"{args.stage.replace('-', '_')}.json")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
