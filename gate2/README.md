# Gate 2：有界窗口纠错基线

本目录把 Refiner 限制为“输入活动源窗口、输出干净窗口文本”的模型适配器。字符
偏移、版本、哈希、校验和提交全部由确定性 Python 组件处理，模型不能直接提交
Patch。

当前实现包括：

- `BoundaryManager`：按可靠标点、endpoint 或 80 字符上限关闭不可变源片段；
- 最近 `K<=3`、总字符 `<=240` 的源窗口与只读前缀；
- Unicode code-point Diff/Patch 编译与精确回放；
- K=1 无证据路径的 `ContentConsistencyChecker`：拒绝源窗口和可信证据均未出现的
  新增正文字符，允许删除、标点和数字格式化；
- mutable tail、修改跨度、敏感数字/否定词、证据租户和状态校验；
- `base_version + base_hash` CAS、过期结果拒绝和 patch hash 幂等；
- 本地约 4B AgenticASR-Refiner 的 Transformers Clean Window 适配器；
- Gate1 `GATE1_GPU1_ROLE=refiner` 模式下真实 Qwen3-ASR partial/final、异步 K=1
  Refiner、版本化 revision 与失败回退链路。

运行确定性测试：

```bash
/home/aim0/data/conda/envs/qwen3-asr/bin/python -m unittest discover -s gate2/tests -v
```

运行 GPU1 重型 Refiner 冒烟：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/aim0/data/conda/envs/qwen3-asr/bin/python -m gate2.run_baseline
```

本地 Refiner 约 4B，只能作为离线/Teacher 基线；它不满足方案中 0.5B～1B 在线
Refiner 的常驻要求。6 条人工场景仅用于管线冒烟，扩大的离线文本基线见下文。

## 2026-09-04 实机冒烟

Refiner 必须使用 AgenticASR 官方 system prompt；将完整结构化任务都塞进 user
消息会导致该 CPM-v2 checkpoint 回显指令并输出 `<KEY>[ASR]`。当前适配器把
`<KEY>[...]` 解析为审计元数据，只把前面的 Clean Window 送给 Diff/Patch。
官方参考实现：<https://github.com/AnXMuy/AgenticASR/blob/main/system/refiner.py>。

最终运行结果位于 `results/baseline_latest.json`：模型加载 1.619 秒，6 个场景平均
推理 0.265 秒；2 个场景提交后改善，0 个场景退化，4 个结果通过校验（其中一个为
不增版本的 `KEEP`）。

| 场景 | 模型结果 | 提交判定 | 结果 |
|---|---|---|---|
| 跨片段时间自我修正 | `会议安排在周四16:00。` | 拒绝：出现无直接依据的 `16/00` 且改动过大 | 回退原文 |
| 地点与可信别名 | `地点在4号楼，就是研发中心那栋。` | 接受：`四→4` 有源文本和租户证据 | CER 改善但非精确匹配 |
| 填充词与重复 | `这个方案，我们后面再讨论。` | 接受 | 精确匹配 |
| 干净直通 | 原样输出 | `KEEP` | 精确匹配且版本不变 |
| 数字格式保护 | `三万元→3万元` | pass-through 无外部证据，保守拒绝 | 安全回退，精确匹配 |
| K=3 多次最终意图 | 只合并标点，未消解历史方案 | 接受但无收益 | 暴露模型能力边界 |

本结果只表示安全管线和真实模型适配器冒烟通过，不表示 Gate2 完成。下一项工作是
建立至少包含 pass-through、自我修正、解释、专名和截断输入的扩大测试集，执行
`SKIP/RULE/K=1/2/3` 对照，并寻找或训练 0.5B～1B 在线 Refiner。

## AASR-Bench 中文离线文本基线

本地数据缓存位于 `/home/aim0/data/datasets/ASR/AASR-Bench`，没有复制到项目。
数据卡没有声明许可证，因此只用于本地研究，不重分发。评测命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/aim0/data/conda/envs/qwen3-asr/bin/python \
-m gate2.aasr_bench_eval \
--batch-size 8 \
--checkpoint-every 40 \
--output-dir gate2/results/aasr_bench
```

2026-09-04 已完成全部 510 条中文 oral-to-clean 文本评测。结果见
`results/aasr_bench/aasr_bench_latest.json`，数据 SHA-256 为
`09ea7a9977b9e6633ca1c8e1cb3f5e717f5c35b5364d8355ba37d92eff92eecc`。

| 指标 | 结果 |
|---|---:|
| 样本 / 参考字符 | 510 / 26,198 |
| 源文本 CER | 35.972% |
| 原始 Refiner CER | 12.425% |
| 安全提交后 CER | 18.868% |
| 原始模型改善 / 退化 | 404 / 42 |
| 提交后改善 / 退化 | 292 / 13 |
| 接受（含 KEEP）/ 拒绝 | 356 / 154 |
| 实际提交改写：改善 / 持平 / 退化 | 292 / 2 / 13 |
| 拒绝候选：改善 / 持平 / 退化 | 112 / 13 / 29 |
| passthrough 原始模型 / 提交后过改率 | 21.053% / 0% |
| batch=8 摊销平均推理时间 | 0.230 s/条 |

安全门把 passthrough 提交过改率从 21.053% 降到 0%，并阻止了 29 条会退化的
候选；但它同时拒绝了 112 条按 CER 计算有改善的候选。拒绝主要由
`unsupported_sensitive_token` 触发（121 条，可与其他原因重叠），说明无证据的
数字/否定词规则安全但偏保守。

13 条已提交退化中，多数是模型删掉后半句、地址、票号或备忘任务；少数是
explanation 场景中“是否应保留拼写解释”的参考口径差异。这说明现有确定性规则只是
结构/证据安全门，不能宣称质量单调改善。简单增加“最小输出长度”也不合适：
真正的填充词和被推翻内容本就需要大幅删除。

模型有 66 条输出达到 `max_new_tokens`：61 条在已完成正文后存在完整或未闭合的
`<KEY>` 审计后缀，适配器剔除后缀后再交给安全门；剩余 5 条没有可靠的元数据分界，
以 `generation_max_tokens` 强制回退。最终正文中 `<KEY>` 泄漏数为 0。

因此这一轮只完成了“AASR-Bench 中文离线确定性文本基线”，不是论文的
rubric Overall，也不是 Qwen3-ASR 端到端结果：本轮没有运行 LLM judge、没有下载或
输入音频，也没有模拟 `K=1/2/3` 流式窗口。下一步是增加内容遗漏风险特征，并在
有明确源片段—输出映射的流式回改集上比较 `SKIP/RULE/K=1/2/3`。

## K=1/2/3 文本流式近似

`streaming_text_eval.py` 参考 AgenticASR 公开的
[`StreamingRefinementSession`](https://github.com/AnXMuy/AgenticASR/blob/main/system/refiner.py)
替换语义：旧 chunk 滑出窗口后单独定稿，最后一个活动区域是最近 K 个源 chunk
的联合重写。运行：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/aim0/data/conda/envs/qwen3-asr/bin/python \
-m gate2.streaming_text_eval \
--batch-size 8 \
--checkpoint-every 256 \
--output-dir gate2/results/streaming_text
```

完整 510 条的修复后结果位于 `results/streaming_text/streaming_text_latest.json`：

| K | 原始 Refiner CER | 安全提交 CER | 原始/提交退化数 | 提交窗口接受率 | 平均原始/提交不稳定度 |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.295% | 17.440% | 28 / 7 | 84.393% | 0.288 / 0.293 |
| 2 | 11.581% | 18.883% | 33 / 8 | 79.548% | 0.350 / 0.327 |
| 3 | 12.845% | 20.666% | 41 / 12 | 74.480% | 0.382 / 0.331 |

237 条样本只产生一个 chunk，三种 K 结果相同。在其余 273 条多 chunk 样本中，
原始 Refiner CER 为 12.195% / 12.611% / 14.447%。以样本为重采样单位做 10,000 次配对
bootstrap：K2-K1 为 +0.416 个百分点，95% CI `[-1.676, 2.209]`，不能认定
K1 与 K2 存在稳定差异；K3-K2 为 +1.836 个百分点，95% CI `[0.358, 3.366]`，
当前 4B checkpoint 在这种分块下 K3 反而显著更差。安全提交的 K2-K1 差值为
+2.097 个百分点，95% CI `[0.867, 3.428]`，因为整窗敏感项校验对大窗口更容易拒绝。

首轮运行曾发现未闭合 `<KEY>[...]` 在 token 上限处被当作正文的适配器错误。
修复后对完整或未闭合审计后缀均会从正文中剔除；无审计后缀且达到 token 上限的
候选强制回退。当前结果中 `<KEY>` 泄漏数为 0。

这是使用完整 oral 文本推导边界的流式近似，没有真实音频的 VAD 时序和 partial
假设演化。它可用于发现窗口替换、遗漏、重复和校验问题，不能当作端到端流式
质量或延迟结论。因此方案中“默认 K=3”应保留为论文复现基线，MVP 默认则先采用
K=1，只在收益预测证明需要时升级到 K=2/3。

### 内容一致性二次校验回归（2026-09-05）

针对真实样本中“不小心”被改成“不动”的负向编辑，K=1 提交前新增
`ContentConsistencyChecker`。该检查器对无证据新增正文字符返回
`ungrounded_content_addition`，由现有回退链保留原始 ASR；删除口癖/重复以及标点
调整不受影响。它目前是无证据流式路径的保守门，未来应由声学、拼音或人工确认等
证据替代，而不是放宽为全局接受。

真实回归结果：S00309 的“其其/也是也是”删除仍通过；自我修正样本第一段删除
“我不想杀他，的不对”通过，第二段错误候选“不小心”→“不动”被拒绝，最终展示
文本保留“不小心”。新增 2 个内容一致性单元测试，Gate2 总计 22 个测试全部通过。
证据文件为 `results/live_integration/20260905_173244/live_smoke.json`（负向回归）和
`results/live_integration/20260905_173249/live_smoke.json`（正向回归）。下一步应在
人工标注集上量化正确纠错被误拒绝的比例与负向编辑率，不能仅凭这两个样本宣告
Gate2 质量完成。

## 35 条真实音频质量评测（2026-09-05）

新增 [`real_audio_manifest.json`](./real_audio_manifest.json) 和
[`real_audio_eval.py`](./real_audio_eval.py)，按 WenetSpeech 本地 JSONL 行号固定
35 条样本，其中 30 条为干净直通控制，5 条覆盖重复、口癖和自我修正。结果见
`results/real_audio_eval/real_audio_eval_latest.json`。

| 指标 | 结果 |
|---|---:|
| 样本数 | 35 |
| 原始 ASR CER | 5.331% |
| K=1 + 二次校验展示 CER | 4.596% |
| 展示文本负向编辑率 | 14.286% |
| 展示文本发生改写比例 | 25.714% |
| 至少一次 Refiner 拒绝比例 | 11.429% |

自我修正子集 CER 从 6.870% 降到 3.053%，但仍有一个候选因
`ungrounded_content_addition` 被拒绝；直通控制子集展示 CER 从 2.556% 上升到
3.834%，负向编辑率为 10%。这说明二次校验已经拦住已知的词汇幻觉，同时当前
Refiner 仍会对干净文本过度改写，Gate2 不能宣称质量单调改善。下一步应扩充并
人工复核该清单，区分参考文本误差与模型负向编辑，并加入拼音/声学证据后重测。

## 扩展评测清单（待人工复核）

`build_real_audio_manifest.py` 可以从本地 WenetSpeech test-net 缓存稳定生成
`real_audio_manifest_expanded.json`。当前清单包含 155 条不同音频：原有 35 条
curated 样本，以及按直通、重复、口癖、自我修正、数字和中英混说各抽取 20 条的
120 条候选。候选记录统一标记为
`annotation_status=heuristic_pending_manual_review`，分类规则只用于覆盖面采样，
不能作为人工真值或论文结论。

先生成并检查清单：

```bash
PYTHONPATH=. python gate2/build_real_audio_manifest.py
```

人工复核并补齐 `expected_clean`、`annotation_status=curated` 后，才运行完整 GPU
评测：

```bash
PYTHONPATH=. python -m gate2.real_audio_eval \
  --manifest gate2/real_audio_manifest_expanded.json \
  --url http://127.0.0.1:8012
```

评测输出会分别报告 `reviewed_sample_count`、
`pending_manual_review_count` 和 `quality_ready`；服务成功运行不等于质量清单已
具备正式验收资格。

### 人工标注表

为避免直接编辑 JSON，可生成 Excel/表格软件可打开的
`real_audio_annotation_sheet.csv`：

```bash
PYTHONPATH=. python gate2/build_annotation_sheet.py
```

对每条候选填写：

- `expected_clean`：听完音频后的标准文本；
- `decision`：只能填 `keep`、`correct` 或 `reject`；
- `final_category`：人工确认后的类别；
- `reviewer_notes`：记录重复、口癖、专名、数字或其他判断。

当完成人工复核后，需将标注回写到 manifest 并将记录标记为
`annotation_status=curated`，再运行正式 GPU 评测。

### 本地音频标注页面

为了在远程机器上直接播放音频，可运行本地标注服务（不使用 GPU）：

```bash
PYTHONPATH=. /home/aim0/data/conda/envs/qwen3-asr/bin/python \
  gate2/annotation_server.py
```

然后在远程桌面的浏览器打开 `http://127.0.0.1:8030`。如果使用 SSH 本地端口转发，将远程
8030 端口转发后在本机打开同一地址。页面只绑定 `127.0.0.1`，不会向公网开放。
页面可以播放当前音频、切换记录、按类别/状态筛选，并在保存后安全回写
`real_audio_annotation_sheet.csv`。

### 生成正式评测清单

人工标注完成后，使用以下命令检查并生成不再依赖启发式类别的正式 manifest：

```bash
PYTHONPATH=. python gate2/curate_annotation_sheet.py --apply
```

它会将 `keep` 统一为 `passthrough`，并将参考文本与
`expected_clean` 存在字词级差异的记录规范为 `correct`。正式清单写入
`real_audio_manifest_curated.json`。之后使用该清单运行评测：

```bash
PYTHONPATH=. python -m gate2.real_audio_eval \
  --manifest gate2/real_audio_manifest_curated.json \
  --url http://127.0.0.1:8012
```

评测输出除按最终类别统计外，还会按 `keep/correct` 分组，且会明确报告被排除样本数。

## 155 条人工复核真实音频评测（2026-09-06）

已完成 155 条不同 WenetSpeech 音频的人工复核并在 GPU0 Qwen3-ASR-1.7B、GPU1
AgenticASR-Refiner 的实际流式服务上重跑。正式评测清单是
`real_audio_manifest_curated.json`，完整证据是
`results/real_audio_eval_curated/20260906_194020/real_audio_eval.json`。所有 155 条均完成，
没有排除样本。

| 分组 | 样本数 | 原始 ASR CER | 展示文本 CER | 负向编辑率 |
|---|---:|---:|---:|---:|
| 全部 | 155 | 11.489% | 12.020% | 21.935% |
| `correct` | 37 | 23.898% | 20.568% | 21.622% |
| `keep` / `passthrough` | 118 | 5.683% | 8.020% | 22.034% |

`correct` 组有收益，但 118 条应当保持原文的样本被过度改写；整体 CER 反而上升
0.531 个百分点。因此本次实验只证明了链路可运行和部分纠错收益，**Gate2 质量验收仍未通过**。下一步应实现选择性纠错，先拦截
`keep` 类低收益窗口，再引入拼音/声学证据改善 `correct` 类的收益和误拒绝。

## Gate3 保守选择性纠错路由（2026-09-06）

新增 `../gate3/rule_router.py`：它只查看在线已获得的 `raw_text`，不使用人工标签、
音频或参考文本。只有出现以下明确语言信号时才会把稳定窗口交给 Refiner：

- 说话人明确收回/重述（如“`不对，我想…`”“说错了”“我的意思”）；
- “不是…，是…”或“不能说…，应该是…”这类前后替换结构；
- 限定的口吃三连字（如“`不不不`”“`死死死死`”）；
- 同一窗口内两个以上功能词的重复开头（如“`其其实也也是`”）。

普通叠词并不触发，例如“拜拜”“好好好吃”“慢慢”“一个个”“战战兢兢”“爸爸妈妈”和
“酸酸甜甜”；数字或英文单独出现也不触发。该规则是可解释的低召回第一版，目的先防止
干净文本的过度改写，而不是替代后续的声学/拼音收益预测器。

离线反事实回放不会启动服务或 GPU：它在路由命中的样本中复用已记录的 Gate2
`visible_text`，未命中样本保留 `raw_text`。运行命令：

```bash
PYTHONPATH=. /home/aim0/data/conda/envs/qwen3-asr/bin/python \
  -m gate3.rule_route_eval \
  --input gate2/results/real_audio_eval_curated/20260906_194020/real_audio_eval.json
```

结果位于 `../gate3/results/rule_route/rule_route_latest.json`：

| 策略 | Refiner 调用率 | 全部 CER | 负向编辑率 | `keep` CER | `keep` 负向编辑率 |
|---|---:|---:|---:|---:|---:|
| 不调用（原始 ASR） | 0.000% | 11.489% | 0.000% | 5.683% | 0.000% |
| 全量调用 Gate2 Refiner | 100.000% | 12.020% | 21.935% | 8.020% | 22.034% |
| Gate3 保守规则路由 | 5.161%（8/155） | **10.990%** | **1.290%** | **5.683%** | **0.000%** |

`correct` 组的调用覆盖率为 18.919%（7/37），CER 从原始 23.898% 降至 22.331%；它没有保留
全量 Refiner 的全部收益（20.568%），但先以极低调用率消除了 `keep` 组的可见退化。这个结果是
同一已标注集的后验回放，不能作为独立泛化结论；下一步需将路由器接入真实异步服务，并用时间上
独立的人工复核音频集重新跑端到端评测。

规则单元测试：

```bash
PYTHONPATH=. /home/aim0/data/conda/envs/qwen3-asr/bin/python \
  -m unittest discover -s gate3/tests -v
```

### Gate3 服务接入与真实音频冒烟（2026-09-06）

`gate1.app` 已在稳定源片段关闭后调用路由器，而不是直接创建 Refiner RPC。默认
`GATE1_REFINER_ROUTER=conservative`；`all` 仅用于复现 Gate2 全量调用基线，`off`
用于审计所有跳过。WebSocket 的 `ready`、`/health` 和会话快照均暴露当前模式；每个
`revision/refiner_keep/refiner_skipped` 事件都携带 `router={call_refiner, score, reasons}`。

真实 GPU 冒烟在临时本地服务上验证了两条路径（服务随后已停止）：

- 干净的 `wenet_00001` 产生 `refiner_skipped`，没有 GPU1 RPC，原始终稿和展示终稿均为
  “我有的时候说不清楚，你们知道吗？”，证据为
  `../gate3/results/live_router/20260906_204315/live_smoke.json`；
- S00309 本次流式 ASR 的等价变体为“其其实也是也是”（不同于离线记录中的“其其实也也是”）。
  新的“功能词重复开头 + 相邻双字词重复”组合规则命中，终稿在 4.322 秒可见，272ms 后收到
  正确 revision“其其实也是”，证据为
  `../gate3/results/live_router/20260906_204722/live_smoke.json`。

为避免 GPU 配额被无意占用，worker 现在各自运行在独立进程组内；服务关闭时会终止整个
进程组，实测不再留下 vLLM `EngineCore` 孤儿进程。上述两条只是功能冒烟，不替代独立
155 条人工复核集的完整端到端复测。

### Gate3 独立留出集：待人工复核

已由 `../gate3/build_holdout_manifest.py` 从同一 WenetSpeech test-net 缓存确定性抽取
50 条新音频，输出为 `../gate3/real_audio_manifest_holdout.json` 及
`../gate3/real_audio_holdout_annotation_sheet.csv`。它按行号排除了正式 155 条清单，脚本
校验结果为重叠 `0`；组成是 35 条直通控制、4 条重复、3 条口癖、3 条自我修正、3 条数字、
2 条中英混说。类别只用于覆盖抽样，不能当作标签。

人工复核服务使用 CPU、不会加载 ASR 或 Refiner：

```bash
PYTHONPATH=. /home/aim0/data/conda/envs/qwen3-asr/bin/python \
  -m gate2.annotation_server \
  --sheet gate3/real_audio_holdout_annotation_sheet.csv \
  --host 127.0.0.1 --port 8030
```

每条听完后保留默认 `keep`（页面会写为 `passthrough`）；若音频证明原始转写需要改正，选择
`correct` 并把 `expected_clean` 改为你认可的最终文本。50 条全部完成后，使用单独的 manifest
固化脚本生成留出真值，再启动一次真实路由服务做端到端验收；此前不得把这批数据用于调整规则。

### Gate3 独立留出集端到端结果（2026-09-06）

50 条均完成复核：34 条 `keep`、16 条 `correct`（其中 3 条由“`keep` 但最终文本实际改变”的
规范化规则转为 `correct`）。保守路由与全量 Refiner 分别用真实 Qwen3-ASR-1.7B +
AgenticASR-Refiner 服务重跑；两个运行的 50 条原始 ASR 终稿逐条一致（不一致数为 0）。

| 策略 | 总 CER | 负向编辑率 | `correct` CER | `keep` CER | `keep` 负向编辑率 |
|---|---:|---:|---:|---:|---:|
| 原始 ASR / 保守路由展示 | **8.819%** | **0.000%** | 13.747% | 5.987% | 0.000% |
| 全量 Refiner | 10.356% | 26.000% | **11.308%** | 9.809% | 29.412% |

补充了评测中的路由审计计数。保守模式实际关闭 61 个源窗口，`router_skips=61`、
`refiner_calls=0`、`refiner_rpcs=0`：它确实完全避免了 GPU1 调用和过改，但在这批独立样本
上也没有召回任何收益窗口。全量模式按配置选择每个关闭窗口；其旧格式评测证据未保存精确 RPC
计数，但质量结果完整。完整对照报告为
`../gate3/results/holdout_comparison/20260906_233525/holdout_comparison.json`。

因此 Gate3 的结论必须修正为：当前规则在正式开发集的后验回放中有净收益，但**未能通过独立留出集
的收益覆盖验收**。不得利用该 50 条继续调阈值或添加例外规则。下一步应保留此 holdout 不动，另建
开发集，以声学置信度、N-best 分歧、拼音/词典证据及文本特征训练并验证收益预测器；再只对固定
holdout 做一次最终验收。

## 真实音频异步集成与并发基线（2026-09-05）

`gate1.app` 现可通过 `GATE1_GPU1_ROLE=refiner` 将 GPU0 固定为
Qwen3-ASR-1.7B 流式 ASR、GPU1 固定为本地约 4B AgenticASR-Refiner。服务先发送
原始 `partial/final`，再对稳定源片段异步执行 K=1 clean-window 重写，经过
Diff/Patch 与校验后发送 `revision/refiner_keep/refiner_reject`，最后发送
`complete`。原始 `raw_text` 与客户端展示 `text` 分开保存。

首轮真实音频发现模型用 `<|im_end|>`（token ID `130073`）结束响应，而旧适配器
只检查 `</s>`（ID `1`），导致正确输出被误报为 `generation_max_tokens`。现在从
模型 `generation_config.json` 读取所有 EOS，并显式加入模板结束 token；同一
S00309 样本从 256-token 假截断变为 13 token 正常停止。

S00309 的完整证据位于
`results/live_integration/20260905_172220/live_smoke.json`。Qwen3-ASR 原始终稿为
“装这个监控，其其实也是也是为了偷窥，一点小利益。”，Refiner 通过两个删除
Patch 生成“装这个监控，其实也是为了偷窥，一点小利益。”；版本从 5 增至 6，
原始终稿约 4.325 秒可见，约 4.600 秒收到 `complete`，纠错额外等待约 274 ms。

`live_concurrency.py` 使用同一 4.45 秒真实音频完成 1/2/4 路同起步阶梯，完整结果
见 `results/live_concurrency/live_concurrency_latest.json`：

| 并发 | 成功会话 | 原始终稿 p95 | complete p95 | 终稿后纠错 p95 | Refiner RPC 最大值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 4.303 s | 4.578 s | 0.275 s | 273.882 ms |
| 2 | 2/2 | 4.351 s | 4.799 s | 0.448 s | 448.077 ms |
| 4 | 4/4 | 4.448 s | 5.300 s | 0.851 s | 850.807 ms |

四路均产生至少一次真实 `revision`，无错误和 OOM。当前 Refiner worker 是串行
JSON-RPC，所以尾延迟随并发近似排队增长；这组单轮阶梯只能证明功能与初始容量，
不能替代三轮重复、p99、30 分钟冒烟或 2～8 小时稳定性测试。

真实自我修正样本
`results/live_integration/20260905_171521/live_smoke.json` 证明第一源片段可正确删除
被推翻的“我不想杀他”；但模型又把后续“不小心”错误改成“不动”，且现有结构/
敏感项校验接受了该 Patch。这是一个已确认的负向编辑反例：Gate2 端到端链路已
打通，但“自我修正稳定改善”和质量二次校验仍未验收，不能据此宣告 Gate2 完成。
