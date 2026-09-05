# ASR 模型在线、离线与流式能力对比

> 调研时间：2026-09  
> 适用项目：双 RTX 3090、多租户、中文优先、在线语音流与离线长音频混合转录、上下文纠错和收益感知调度。

## 1. 先区分三个概念

“在线转录”不一定等于“原生流式”。一个离线模型也可以被 HTTP/WebSocket 服务在线调用，但如果每次都需要拿到完整音频或较大的滑动窗口后重新推理，它仍然不是严格意义上的增量流式模型。

| 概念 | 本文含义 | 评价重点 |
|---|---|---|
| 在线转录 | 服务在音频上传过程中持续返回中间结果，或通过 WebSocket 提供实时服务 | 首字延迟、端点到终稿延迟、断流恢复、并发能力 |
| 离线转录 | 音频文件完整到达后再进行一次或多次推理 | CER/WER、长音频稳定性、吞吐、时间戳 |
| 原生流式 | 模型或官方推理 SDK 维护增量状态，只处理新增音频块 | 状态语义、重复计算、窗口边界、字幕稳定性 |

因此，`Whisper-Streaming` 可以提供在线流式服务，但 `Whisper` 模型本身仍然主要是离线/窗口式推理；`Qwen3-ASR` 的官方 vLLM streaming state 则属于更明确的原生流式接口。

## 2. 总体对比表

下表中的“可用”表示适合进入本项目的实验或工程链路，不表示已经在双 RTX 3090 上实测达标。所有延迟、显存和并发结论都必须通过同一数据集、同一硬件和同一压测脚本复现。

| 模型/实现 | 在线转录 | 离线转录 | 流式性质 | 批处理/时间戳 | 对本项目的可用性 | 建议角色 |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 官方 vLLM streaming state；可封装 WebSocket | Transformers/vLLM 支持离线推理和 batch | **原生流式**：`init_streaming_state`、`streaming_transcribe`、`finish_streaming_transcribe` | 官方流式不支持 batch、时间戳和 ForcedAligner；离线时间戳使用独立 `Qwen3-ForcedAligner-0.6B` | **高**。适合作为论文主线，但单卡活跃会话数必须实测 | 论文主模型、质量主线 |
| **Qwen3-ASR-0.6B** | 同一官方 vLLM streaming state | 支持离线推理；适合容量和速度对照 | **原生流式** | 与 1.7B 相同，流式时间戳能力受限 | **高**。可作为 GPU0 容量备选或轻量消融 | 轻量流式备选 |
| **Fun-ASR-Nano-2512** | FunASR 官方提供 vLLM streaming SDK 和 WebSocket 服务 | 支持 FunASR 推理、vLLM batch 和长音频分段 | **原生/官方 SDK 流式**；示例使用 `FunASRNanoStreamingVLLM`，默认示例 chunk 为 720 ms | 具备 vLLM batch；时间戳、置信度和版本兼容性需按具体 checkpoint 实测 | **高**。是最值得优先验证的同类 LLM-ASR 流式备选 | Qwen3-ASR 的同类流式对照、工程备选 |
| **Fun-ASR-MLT-Nano-2512** | 可复用 FunASR/vLLM streaming 路径 | 支持离线和 batch | **官方 SDK 流式** | 面向 31 语言，中文结果必须单独测量 | **中**。适合多语种扩展，不宜替代中文主线 | 多语种扩展 |
| **Paraformer-zh-streaming** | FunASR runtime 可提供 WebSocket 在线服务 | Paraformer 离线模型支持文件/批量转录 | **原生流式模型**，通常配合 VAD、标点和 2-pass | 运行时支持 online/offline/2-pass、VAD、标点和热词；时间戳取决于模型和配置 | **高**。中文低延迟和高并发的成熟工程基线 | 低延迟工程对照 |
| **sherpa-onnx Zipformer / SenseVoice / Paraformer** | Python/C++/WebSocket 等在线服务方式成熟 | 支持 ONNX 离线转录 | 多数候选提供**原生在线状态**；模型能力取决于具体 checkpoint | 轻量部署，时间戳和置信度能力因模型而异 | **高**，尤其适合 CPU/轻量高并发；不直接代表 LLM-ASR 质量 | 轻量实时基线、AgenticASR 前端 |
| **faster-whisper** | 可通过服务封装在线接口；实时需外加窗口策略 | **强**：GPU FP16/INT8、batch、VAD、词级时间戳 | 模型本身不是原生增量流式；通常配合 Whisper-Streaming 或 WhisperLive | 支持 batch、VAD、词级时间戳 | **中高**。适合离线质量、多语种和时间戳对照；在线成本需实测 | 离线基线、时间戳基线 |
| **whisper.cpp** | 提供 HTTP 服务和实时麦克风示例 | 支持量化、CPU/CUDA 和离线文件转录 | 实时示例主要是重复窗口推理，不能等同于原生流式 | 支持量化和端侧部署；时间戳能力依参数和封装而定 | **中**。适合低资源/端侧对照，不适合作为主服务后端 | 量化和端侧基线 |
| **FireRedASR2-AED / FireRedASR2-LLM** | 可被封装成服务；公开主路径偏 batch | **强**：普通话、20+ 方言/口音、英文和中英混说；AED 支持词级时间戳与置信度 | ASR 主路径不能直接视为 Qwen3-ASR 式原生流式；FireRedVAD 另有流式能力 | batch、时间戳和置信度能力较完整，具体以模型卡为准 | **中高**。非常适合中文方言/困难音频离线质量对照 | 质量上限和困难音频基线 |
| **GLM-ASR-Nano-2512** | 可通过 Transformers/SGLang 服务化 | **强**：适合离线困难音频、低音量和多语种对照 | 暂未确认有与 Qwen3-ASR 等价的官方 streaming state | 具体 batch、时间戳和服务接口以官方版本为准 | **中**。可做离线质量对照，不建议直接放在线主链路 | 困难音频离线基线 |
| **WeNet Conformer/Paraformer** | 官方 runtime 支持流式服务 | 支持非流式/离线推理 | **原生流式工具链**，含状态缓存和多种 runtime | GPU、ONNX、TensorRT 等部署路径；模型和配置差异较大 | **中高**。学术可解释性好，但集成工作量高 | 学术流式对照 |
| **NVIDIA Nemotron-3.5-ASR-Streaming-0.6B** | 官方 NeMo 提供 cache-aware streaming 路径 | 支持非流式或统一模型路径，具体以模型卡为准 | **原生流式**，官方宣传可控约 80 ms～1 s 延迟 | 多语种；中文支持和 3090 兼容性需先核对模型卡 | **待验证**。确认中文后再进入多语种实验 | 多语种低延迟扩展 |
| **Omnilingual ASR** | 可服务化，但官方参考路径偏 batch | 支持 CTC/LLM-ASR batch，覆盖 1600+ 语言 | 不是本项目优先的原生流式方案 | 300M/1B/3B/7B 多种规模；官方 README 对单段长度有约束 | **中低**。适合极端多语种离线研究，不适合中文实时主链路 | 多语种研究扩展 |

## 3. 按项目任务的推荐

### 3.1 在线实时转录

优先顺序建议如下：

1. `Qwen3-ASR-1.7B`：研究主线，验证上下文纠错和收益感知调度。
2. `Fun-ASR-Nano-2512`：同类 LLM-ASR 流式备选，优先验证其官方 SDK 与 vLLM 兼容性。
3. `Paraformer-zh-streaming + FunASR 2-pass`：低延迟、VAD、标点、热词和多客户端工程基线。
4. `sherpa-onnx Zipformer`：CPU/轻量高并发基线，也适合复现 AgenticASR 的在线 ASR 前端。
5. `Whisper-Streaming + faster-whisper`：作为实时对照，不建议直接当作最低延迟方案。

### 3.2 离线长音频转录

- 论文主质量路径：`Qwen3-ASR-1.7B` 离线 batch。
- 中文方言和困难音频：`FireRedASR2-AED/LLM`、`GLM-ASR-Nano-2512`。
- 多语种和时间戳：`faster-whisper large-v3/turbo`。
- 大规模吞吐：`Fun-ASR-Nano-2512` vLLM batch 或 FunASR 的分段批处理。
- 精确时间戳：优先使用模型原生时间戳，或使用与 ASR 结果匹配的独立 ForcedAligner；不要把在线近似时间和离线精确时间混为同一指标。

### 3.3 多租户调度实验

为了证明调度算法而不是模型差异，建议只选一个在线后端进入完整链路：

- 论文主线固定为 `Qwen3-ASR-1.7B`；
- 替代后端从 `Fun-ASR-Nano-2512`、`Paraformer-zh-streaming`、`sherpa-onnx Zipformer` 中三选一；
- `faster-whisper`、`FireRedASR2`、`GLM-ASR` 只做离线质量或时间戳基线；
- 不要在 FIFO、EDF、HRRN 和收益感知调度实验中更换 ASR 后端。

## 4. 对本项目的最终建议

### 推荐的主线组合

```text
GPU 0：Qwen3-ASR-1.7B 原生流式
GPU 1：离线 ASR、Refiner、ForcedAligner 的仲裁执行
CPU：音频解码、重采样、VAD 前处理、队列和 WebSocket
```

### 推荐的替代组合

若 Qwen3-ASR-1.7B 在目标并发下无法满足首字延迟或 RTF：

- 首先测试 `Qwen3-ASR-0.6B`，保持研究变量最少；
- 若需要同类 LLM-ASR 的工程替代，测试 `Fun-ASR-Nano-2512`；
- 若更看重低延迟和并发，测试 `Paraformer-zh-streaming + 2-pass`；
- 若更看重 CPU/边缘和高并发，测试 `sherpa-onnx Zipformer`。

### Whisper 的明确定位

Whisper 可以使用，但应在方案中写清楚使用方式：

| 使用方式 | 适合度 | 原因 |
|---|---|---|
| `faster-whisper` 离线转录 | 高 | 质量、多语种、batch、VAD 和词级时间戳较完整 |
| `Whisper-Streaming + faster-whisper` 在线转录 | 中高 | 可以实时返回，但需要滑动窗口、重叠和稳定前缀算法 |
| `WhisperLive` 在线服务 | 中 | 便于快速搭建演示和基线，但仍受重复窗口推理影响 |
| `whisper.cpp` 端侧/量化 | 中高 | 资源占用小，适合端侧和量化实验 |
| Whisper 作为本项目默认中文低延迟主后端 | 中低 | 原生模型没有增量状态，在线多租户下重复计算成本较高 |

## 5. 统一实验规范

不同模型只有在以下条件固定后才能比较：

- 相同音频集合、采样率、声道和文本规范化规则；
- 相同 VAD 或明确记录 VAD 差异；
- 相同在线 chunk 时长、窗口大小、重叠比例和端点规则；
- 相同硬件、精度、量化方式、CUDA 和推理框架版本；
- 相同并发注入方式和音频实时倍率；
- 分别报告 `TTFP`、`EOU-to-Final`、`RTF`、P95/P99、CER/WER、峰值显存和峰值内存；
- 在线模型与离线模型分开报告，不能用离线 CER 直接证明在线流式质量；
- 调度算法实验固定 ASR 后端，只改变调度策略；
- Whisper 的滑动窗口重复计算应单独统计每秒音频对应的实际推理秒数。

建议的最小矩阵：

| 实验组 | 固定项 | 变化项 | 目的 |
|---|---|---|---|
| 模型对比 | 硬件、音频、评测脚本、VAD | ASR 模型/后端 | 选择主线和工程备选 |
| 流式策略对比 | ASR 模型、硬件、音频 | 原生 state、滑动窗口、2-pass | 评估延迟、抖动和重复计算 |
| 调度对比 | ASR 后端、Refiner、chunk、硬件 | FIFO、EDF、HRRN、收益感知 | 证明调度算法收益 |
| 离线质量对比 | 音频、文本规范化、batch 策略 | Qwen、Fun-ASR、FireRed、GLM、Whisper | 建立质量和困难场景基线 |

## 6. 结论

本项目不需要把所有模型都建设成生产链路。最合理的硕士课题范围是：

1. `Qwen3-ASR-1.7B` 作为论文主线；
2. 从 `Fun-ASR-Nano-2512`、`Paraformer-zh-streaming`、`sherpa-onnx Zipformer` 中选一个完成工程替代；
3. `faster-whisper` 作为离线、多语种和时间戳基线；
4. `FireRedASR2` 或 `GLM-ASR` 选一个作为中文困难音频质量对照；
5. Whisper 的在线方案只用于流式策略对比，不把它的重复窗口结果包装成原生流式能力。

这样既能覆盖在线、离线和流式转录，也能保证模型比较、调度比较和纠错比较之间的实验变量清晰。

## 7. 参考资料

- Qwen3-ASR：<https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR vLLM streaming 示例：<https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py>
- Fun-ASR：<https://github.com/QwenAudio/Fun-ASR>
- Fun-ASR-Nano-2512：<https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512>
- FunASR：<https://github.com/modelscope/FunASR>
- FunASR Paraformer streaming 示例：<https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/paraformer_streaming>
- sherpa-onnx：<https://github.com/k2-fsa/sherpa-onnx>
- FireRedASR：<https://github.com/FireRedTeam/FireRedASR>
- FireRedASR2S：<https://github.com/FireRedTeam/FireRedASR2S>
- GLM-ASR：<https://github.com/zai-org/GLM-ASR>
- WeNet：<https://github.com/wenet-e2e/wenet>
- faster-whisper：<https://github.com/SYSTRAN/faster-whisper>
- Whisper-Streaming：<https://github.com/ufal/whisper_streaming>
- WhisperLive：<https://github.com/collabora/WhisperLive>
- whisper.cpp：<https://github.com/ggml-org/whisper.cpp>
- NVIDIA NeMo/Speech：<https://github.com/NVIDIA-NeMo/Speech>
- Omnilingual ASR：<https://github.com/facebookresearch/omnilingual-asr>
