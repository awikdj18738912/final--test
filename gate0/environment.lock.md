# Gate 0 tested environment

This is the environment actually used for the Gate 0 run on 2026-09-04.
The machine has two independent NVIDIA GeForce RTX 3090 GPUs (24 GiB each).
The model paths are local and therefore do not depend on a network download.

| Component | Tested value |
|---|---|
| OS / kernel | Linux / 6.17.0-35-generic |
| Python | 3.12.12 (Conda environment `qwen3-asr`) |
| NVIDIA driver | 580.159.04 |
| CUDA reported by PyTorch | 12.8 |
| PyTorch | 2.9.1+cu128 |
| torchaudio | 2.9.1 |
| qwen-asr | 0.0.6 |
| vLLM | 0.14.0 |
| transformers | 4.57.6 |
| NumPy | 2.2.6 |
| soundfile | 0.13.1 |
| FastAPI | 0.133.1 |
| Uvicorn | 0.41.0 |
| websockets | 16.0 |

Local model artifacts used:

- `/home/aim0/data/models/ASR/Qwen3-ASR-1.7B`
- `/home/aim0/data/models/ASR/Qwen3-ASR-0.6B`
- `/home/aim0/data/models/ASR/AgenticASR-Refiner`

The exact machine-readable package and hardware snapshot is stored in
`gate0/results/latest.json`. Re-run the preflight after any package, driver,
model, or kernel change.
