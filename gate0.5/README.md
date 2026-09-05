# Gate 0.5：Paraformer online 对照

本目录保存用于后端选型的可复跑 FunASR Paraformer online 基线。该后端使用独立的
`para` Conda 环境，不与现有 `qwen3-asr` 环境混装：

```text
Python       3.10.20
FunASR       1.2.9
ModelScope   1.34.0
PyTorch      2.5.1
Torchaudio   2.5.1
```

模型为
`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`，本地权重位于
`/home/aim0/data/models/funasr/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`。
脚本只加载一个模型，每个会话持有独立 online cache，并由单 worker 轮询送入会话帧；
它不使用多个 Python 线程并发调用同一个有状态 CUDA 模型。

## 可复跑命令

下面的正式命令使用与 Gate1 Qwen3-ASR-1.7B 压测相同音频的前 5 秒。由于原始
AISHELL 标注覆盖完整 14.7 秒，截断片段的参考文本通过 `--reference-text` 明确给出：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/aim0/anaconda3/envs/para/bin/python \
gate0.5/paraformer_benchmark.py \
  --model-path /home/aim0/data/models/funasr/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --audio /home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell/wav/test/S0768/BAC009S0768W0452.wav \
  --max-audio-sec 5 \
  --reference-text '白云区钟落潭竹一村民' \
  --repetitions 3 \
  --concurrency 1,2,4 \
  --output-dir gate0.5/results
```

`--repetitions 3` 同时作用于 buffered、paced 和每个并发级别。结果写入带时间戳的
目录，并同步到 `gate0.5/results/paraformer_online_latest.json`。

Qwen3-ASR-0.6B 复用 Gate1 服务和压测器，使用独立端口与运行目录：

```bash
GATE1_RUNTIME_DIR=/tmp/asr-gate1-qwen06 \
GATE1_REALTIME_MODEL=/home/aim0/data/models/ASR/Qwen3-ASR-0.6B \
GATE1_OFFLINE_MODEL=/home/aim0/data/models/ASR/Qwen3-ASR-0.6B \
/home/aim0/data/conda/envs/qwen3-asr/bin/python \
  -m uvicorn gate1.app:app --host 127.0.0.1 --port 8001

/home/aim0/data/conda/envs/qwen3-asr/bin/python -m gate1.load_test \
  --url http://127.0.0.1:8001 --audio-sec 5 --frame-ms 250 \
  --concurrency 1 2 4 --repetitions 3 \
  --output-dir gate0.5/results/qwen06
```

## 2026-09-04 实机结果

测试硬件为 RTX 3090 GPU 0，输入为 16 kHz 单声道、5 秒音频。配置为 600 ms
输入帧、`chunk_size=[0,10,5]`、encoder look-back 4、decoder look-back 1。

| 并发会话 | 成功会话 | TTFP p50 | TTFP p95 | TTFP p99 | 最终结果 p95 | CER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 0.6599 s | 0.6603 s | 0.6604 s | 4.8620 s | 0.20 |
| 2 | 6/6 | 0.6878 s | 0.7189 s | 0.7190 s | 4.9273 s | 0.20 |
| 4 | 12/12 | 0.7482 s | 0.8351 s | 0.8355 s | 5.0507 s | 0.20 |

其他指标：模型加载 `2.8421 s`，稳态 buffered compute RTF 均值 `0.1015`、p95
`0.1017`，峰值显存 `1.235 GiB`。假设文本为“白云区中洛潭竹一村民”；参考为
“白云区钟落潭竹一村民”。

与 Gate1 的 Qwen3-ASR-1.7B 结果相比，同一输入和同样的 1/2/4 路会话规模下：

| 后端 | 1 路 TTFP p95 | 2 路 TTFP p95 | 4 路 TTFP p95 | 该截断句 CER | 已观测峰值显存 |
|---|---:|---:|---:|---:|---:|
| Qwen3-ASR-1.7B | 0.8257 s | 0.7995 s | 0.8440 s | 0.10 | 17.124 GiB |
| Qwen3-ASR-0.6B | 0.8165 s | 0.7833 s | 0.8082 s | 0.10 | 17.054 GiB |
| Paraformer online | 0.6603 s | 0.7189 s | 0.8351 s | 0.20 | 1.235 GiB |

这里的模型原生流式参数不同：Qwen 服务接收 250 ms 客户端帧、服务端使用 1 秒
chunk；Paraformer 使用 600 ms 帧。因此该表适合比较各自推荐配置下的工程表现，
不能归因于单一模型结构。0.6B 正式归档轮次的单路 p95 为 0.8165 秒；
在前一次独立服务启动中曾出现一次 1.2576 秒的首请求离群值，需在 soak test
继续观测冷形状/编译缓存波动。CER 也只有一个截断句，不构成质量排名。

Qwen3-ASR-0.6B 的完整结果位于 `gate0.5/results/qwen06/load_test_latest.json`：

| 并发会话 | 成功会话 | TTFP p95 | TTFP p99 | 最终结果 p95 | CER |
|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 0.8165 s | 0.8207 s | 4.7929 s | 0.10 |
| 2 | 6/6 | 0.7833 s | 0.7835 s | 4.8015 s | 0.10 |
| 4 | 12/12 | 0.8082 s | 0.8087 s | 4.8414 s | 0.10 |

## 当前判定

- Paraformer online 已通过 1/2/4 路功能与初始延迟门槛，可作为低显存、低延迟的
  工程候选后端。
- Qwen3-ASR-1.7B 的 Gate1 结果同样满足当前 TTFP 门槛，且单句 CER 更低，所以
  当前不切换 MVP 默认后端；论文主线仍保留 Qwen3-ASR。
- Qwen3-ASR-0.6B 的同条件 1/2/4 路矩阵已经补齐；它可作为 1.7B 的容量降级后端，
  但当前 vLLM 显存池配置下没有带来明显显存下降，不能只凭参数量假设节省显存。
- 上述单句结果只用于验证并发评测链路；最终质量判断以接下来的 200 条统一评测为准。
- 本结果只测试 Paraformer online checkpoint，不代表 FunASR 2-pass、offline、VAD、
  标点或时间戳链路已经验证。后续 2-pass 结果必须分别记录这些组件。

## 200 条分层质量评测

固定清单见 `quality_manifest_200.json`，由 `build_quality_manifest.py` 从 AISHELL-1
test 的 7,176 条音频中以种子 `20260904` 生成。清单包含 20 位说话人、200 条音频、
总时长 1,263.51 秒：`<5 s`、`5～10 s`、`>=10 s` 分别为 80、80、40 条。长音频在
原始 test 集中仅有 71 条，本清单有意过采样 40 条作为压力层，因此总体 CER 是受控
后端比较指标，不是 AISHELL-1 test 全集的总体性能估计。

三组均采用流式状态进行 buffered quality decode；计算 RTF 只描述吞吐，不替代前面
paced 测得的客户端 TTFP。中文主指标为去除空白和标点后的语料级 CER。没有选择
固定分词器前不报告 WER，避免将分词差异误算为识别差异。

| 后端 | 成功 | 语料级 CER | 宏平均 CER | 短音频 CER | 中音频 CER | 长音频 CER | Compute RTF mean / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-ASR-1.7B | 200/200 | 2.561% | 2.333% | 1.684% | 1.532% | 5.300% | 0.0610 / 0.0686 |
| Qwen3-ASR-0.6B | 200/200 | 3.796% | 3.408% | 1.895% | 2.865% | 7.489% | 0.0388 / 0.0430 |
| Paraformer online | 200/200 | 4.700% | 4.206% | 2.632% | 3.664% | 8.756% | 0.0988 / 0.1061 |

逐句配对 bootstrap 使用 10,000 次重采样、固定种子 `20260904`。CER 差值为
`A-B`，负值表示 A 更好：

| 配对 | A胜 / 平 / B胜 | CER 差值 | 95% bootstrap CI |
|---|---:|---:|---:|
| 1.7B vs 0.6B | 29 / 161 / 10 | -1.235 pct-pt | [-1.904, -0.630] pct-pt |
| 1.7B vs Paraformer | 53 / 133 / 14 | -2.139 pct-pt | [-3.021, -1.313] pct-pt |
| 0.6B vs Paraformer | 39 / 139 / 22 | -0.904 pct-pt | [-1.709, -0.123] pct-pt |

原始结果和比较证据：

- `results/quality/qwen17/qwen-streaming_latest.json`
- `results/quality/qwen06/qwen-streaming_latest.json`
- `results/quality/para/paraformer-online_latest.json`
- `results/quality/quality_comparison.json`

### Gate 0.5 最终基线结论

在当前 AISHELL-1 普通话朗读语音范围内，Qwen3-ASR-1.7B 的 CER 显著低于另外两个
后端，且已在 Gate1 满足 1/2/4 路初始延迟门槛，因此冻结为论文主线和 MVP 默认
在线后端。Qwen3-ASR-0.6B 吞吐最快，冻结为过载容量降级后端。Paraformer online
仍保留为低显存后备和非 LLM 工程对照，但不替换默认后端。

Gate0.5 在“普通话朗读语音基线选型”范围内通过。噪声、方言、中英混说、领域专名、
真实会议以及 FunASR 2-pass 属于后续外部有效性与终稿链路实验，不能从本结果外推。
