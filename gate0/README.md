# Gate 0: dual RTX 3090 baseline

This directory contains the first executable milestone for the ASR project. It
uses the existing `qwen3-asr` environment and local model weights. The runner
keeps the topology deliberately simple:

- GPU 0: one Qwen3-ASR vLLM streaming process.
- GPU 1: one Qwen3-ASR vLLM offline process followed by one Transformers
  Refiner process. They are separate child processes and never overlap; this is
  the initial serialized GPU1 Arbiter.
- Training: both service processes must be stopped before both GPUs are used.

## Run

The environment can be addressed by its absolute Python path, so activation is
not required:

```bash
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/run_gate0.py --stage all
```

Use another 16 kHz audio file with `--audio`. To run one stage only:

```bash
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/run_gate0.py --stage preflight
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/run_gate0.py --stage realtime --gpu 0
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/run_gate0.py --stage offline --gpu 1 --gpu-memory-utilization 0.50
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/run_gate0.py --stage refiner --gpu 1
/home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/training_mode.py --check
```

`all` writes a timestamped directory and `gate0/results/latest.json`. Each
model stage also has a `.log` file. The recorded fields include model load
time, inference time, TTFP, RTF, language/text, and sampled GPU memory peak.

The tested versions are listed in `environment.lock.md`. The current formal
run is retained under `results/20260904_123622/`; `results/latest.json` points
to the same summary.

`max_model_len=8192` is intentional. vLLM's default 65536 context requires
more KV cache than the conservative test budget and is unnecessary for this
first speech baseline. vLLM stages must be started from this Python file (not
from stdin) because its worker uses Python `spawn`.

The checked-in local Refiner is `AgenticASR-Refiner`, a roughly 4B Llama model.
It is therefore measured as an offline/heavy Refiner baseline and is not
claimed to be the 0.5B--1B online Refiner from the research plan. A small local
Refiner can be substituted with `--refiner-model` after it is installed.

`all` reports functional success separately from the initial performance
targets. A result of `pass_with_capacity_boundary` means the environment and
all stages ran, but the measured realtime TTFP/RTF exceeded the provisional
Gate0 threshold; use the 0.6B model or a later backend profile for the online
capacity path.

## Warm streaming capacity matrix

The single-stage Gate0 check includes model load time and a short utterance.
Use the following capacity benchmark for the next decision point. It loads
each Qwen model once, warms every parameter variant, and records three warm
samples on a 14.7-second AISHELL utterance:

```bash
CUDA_VISIBLE_DEVICES=0 /home/aim0/data/conda/envs/qwen3-asr/bin/python gate0/capacity_benchmark.py
```

The matrix is in `capacity_matrix.json`. `compute_rtf` is measured with the
input buffer delivered immediately; `e2e_ttfp_sec` instead paces audio input
and is the client-visible latency. The latter necessarily includes enough
audio for the ASR chunk, so it is not interchangeable with decoder compute
time. Output is stored in `results/capacity/<timestamp>/capacity.json` and
mirrored to `results/capacity_latest.json`.
