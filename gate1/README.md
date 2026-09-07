# Gate 1: dual-process ASR service baseline

This is the first runnable service built from the Gate0 profile. The API
process does not import an ASR model. At startup it launches two child
processes with fixed GPU visibility. GPU1 has two mutually exclusive roles:

- GPU0: one Qwen3-ASR realtime streaming worker, using the tested `1.0 s`
  server chunk and `0.25 s` client frame recommendation.
- `GATE1_GPU1_ROLE=offline`（默认）：one independent Qwen3-ASR offline worker.
- `GATE1_GPU1_ROLE=refiner`：one local AgenticASR-Refiner worker for asynchronous
  Gate2 K=1 clean-window revisions; the offline API returns 503 in this mode.
  `GATE1_REFINER_ROUTER=conservative` is the default: only Gate3 high-evidence
  windows are submitted. Use `all` only to reproduce the previous full-call
  baseline, or `off` to emit audited skips for every closed span.

The initial storage is in-memory. It is intentionally a Gate1 baseline, not a
restart-safe production system: sessions, jobs, and idempotency records are
lost when the API process stops. The worker boundary is retained so Redis,
durable jobs, GPU1 arbitration, and a second ASR backend can be added without
placing CUDA work in FastAPI's event loop.

Each GPU worker runs in a separate process group. Stopping the service signals
the entire group, so vLLM's `EngineCore` child cannot remain orphaned and hold
GPU memory after its wrapper exits.

## Start

Run this from the repository root. Startup loads a Qwen model on each GPU and
can take about a minute.

```bash
/home/aim0/data/conda/envs/qwen3-asr/bin/python -m uvicorn gate1.app:app --host 127.0.0.1 --port 8000
```

启动完成后直接打开 `http://127.0.0.1:8000/` 使用内置前端。页面支持浏览器麦克风实时流式
转写、Qwen 原文与 Refiner 修订对照、事件时间线、Gate3 流式遥测，以及完整音频文件的非流式
上传。麦克风功能要求浏览器从 localhost 或 HTTPS 访问；远程服务器应先建立 SSH 端口转发。
非流式上传只在 `GATE1_GPU1_ROLE=offline` 时可用，Refiner 模式下页面会显示相应提示。

The service owns `gate1/runtime/realtime.sock`, `offline.sock`, worker logs,
and temporary uploaded audio. Stop Uvicorn to terminate both worker children.
Use a different runtime location when running multiple instances:

```bash
GATE1_RUNTIME_DIR=/tmp/asr-gate1-alt /home/aim0/data/conda/envs/qwen3-asr/bin/python -m uvicorn gate1.app:app --port 8001
```

启动 GPU1 Refiner 模式：

```bash
GATE1_GPU1_ROLE=refiner \
GATE1_RUNTIME_DIR=/tmp/gate2-live-integration \
/home/aim0/data/conda/envs/qwen3-asr/bin/python \
-m uvicorn gate1.app:app --host 127.0.0.1 --port 8012
```

Each worker completes a real local ASR warm-up before its Unix socket is
created, so the health endpoint only becomes available after compilation and
first-inference costs have been paid. Its reported `warmup_sec` is a startup
cost, not a client latency metric.

## API smoke test

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: demo-session-1' \
  -d '{"tenant_id":"demo","mode":"realtime","language":"Chinese"}'
curl -X POST http://127.0.0.1:8000/offline/jobs \
  -H 'Idempotency-Key: demo-offline-1' \
  -F tenant_id=demo -F language=Chinese \
  -F file=@/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0768/BAC009S0768W0452.wav
```

Poll an offline job using `GET /offline/jobs/{job_id}?tenant_id=demo`; cancel it
using `POST /offline/jobs/{job_id}/cancel?tenant_id=demo`.

For realtime transcription, first create a session, then connect to
`ws://127.0.0.1:8000/sessions/{session_id}/stream?tenant_id=demo`. After the
server sends `ready`, send 16 kHz mono PCM signed-16-bit little-endian binary
frames (recommended duration: 250 ms). The server sends versioned `partial`
events whenever its cumulative ASR hypothesis changes. Send `{"event":"end"}`
to receive a `final` event. JSON audio frames are also supported when they
provide a strictly increasing `sequence_id`, `sample_rate: 16000`, and
base64-encoded `pcm16_b64`.

在 Refiner 模式中，`partial/final` 先返回 Qwen3-ASR 原始累计假设，稳定源片段在
GPU1 异步生成 clean window；校验通过后发送版本化 `revision`，无修改发送
`refiner_keep`，被 Gate3 路由器跳过时发送带规则分数/原因的 `refiner_skipped`；拒绝或
异常发送 `refiner_reject/refiner_error` 并保留原文，最后以 `complete` 关闭会话。
`raw_text` 始终保留原始 ASR 文本，`text` 是客户端当前展示文本。

每次 ASR 累计假设更新还会携带 `asr_stream_metrics`：已处理 chunk 数、假设更新次数、被后续
假设替换的前缀字符数、追加字符数及不稳定度。它们只描述在线已经观察到的文本演化，不改变
Qwen3-ASR 解码或客户端展示；会话最终快照保留这些指标，供后续 Gate3 收益预测训练使用。

开发集采集可额外设置 `GATE1_COLLECT_LOGPROB_TELEMETRY=1`。此开关默认关闭；开启时以相同的
贪婪解码参数请求 vLLM 返回 logprob，但仅在 `asr_stream_metrics` 输出当前流最后一次解码的标量
摘要（`mean_token_logprob`、`min_token_logprob`、token 数），不会输出 token、候选词或原始概率表。
它尚不是置信度阈值，也不能直接用于生产路由；仅可用于新的 Gate3 开发集特征采集。

真实音频单路与 1/2/4 路阶梯测试：

```bash
/home/aim0/data/conda/envs/qwen3-asr/bin/python -m gate2.live_smoke \
  --url http://127.0.0.1:8012 --audio /path/to/audio.wav

/home/aim0/data/conda/envs/qwen3-asr/bin/python -m gate2.live_concurrency \
  --url http://127.0.0.1:8012 --audio /path/to/audio.wav --levels 1 2 4
```

## Current limits

- One realtime worker serializes state updates. Run the reproducible paced
  ASR-only probe after startup:

  ```bash
  /home/aim0/data/conda/envs/qwen3-asr/bin/python -m gate1.load_test --url http://127.0.0.1:8000
  ```

It sends a shared 5-second sample in 250 ms PCM frames and writes p50/p95
first-result and final-result latency to `gate1/results/`.

The observed 2026-09-04 run completed all 21 sessions (3 repetitions each at
1, 2, and 4 concurrent sessions). First-result p95 was 0.826 s, 0.800 s, and
0.844 s respectively; final-result p95 was 4.815 s, 4.833 s, and 4.908 s.
The raw run is `results/20260904_134819/load_test.json`.
- Offline cancellation is cooperative: a queued job is cancelled immediately;
  a running single-model inference finishes its current call, then its result
  is discarded.
- Refiner 模式当前使用约 4B checkpoint，并把请求在单个 GPU1 worker 中串行执行；
  它是集成基线，不是方案最终要求的 0.5B～1B 在线模型，也尚未实现与 offline ASR
  的 batch-boundary arbiter。
- No VAD, persistence, durable queue, trusted-memory retrieval, GPU1
  batch-boundary arbiter, or backend fallback is included yet.
