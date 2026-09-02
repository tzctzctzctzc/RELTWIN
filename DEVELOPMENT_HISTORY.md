# SpotSound / RelTwin / SetPO / NOVA / SpanTool 超详细开发与实验交接文档

> 文档状态：2026-09-02 冻结快照  
> 目标会议：ICASSP 2027 方向储备  
> GitHub：<https://github.com/tzcinhust/spotsound>  
> 当前主开发分支：`codex/spantool-full`  
> 当前可公开主结果：NOVA 在 SpotSound-Bench 上的“开放额外数据、重建同评测器、点估计 SOTA”  
> 当前研究前沿：SpanTool 已完整实现并审计，但尚未在公开 SpotSound/Clotho 上超过当前最强结果

这份文档不是只描述最终代码，而是完整记录项目为什么这样选、每一轮实验解决了什么问题、哪些结果真的成立、哪些尝试失败、服务器上有什么、队友如何复现，以及后续最值得投入的路线。它应与仓库中的原始 JSON 证据、交接压缩包中的脱敏聊天记录和服务器实验元数据一起阅读。

## 0. 一页结论：接手前必须知道的事实

### 0.1 当前最强结论

1. **SpotSound-Bench 当前锁定方法是 E004 NOVA**：mIoU `59.329`、R1@0.3 `78.75`、R1@0.5 `62.25`。它高于论文表中 SpotSound-A 的 `57.9 / 76.5 / 59.5`，也高于我们用同一重建评测器复测官方 checkpoint 的 `58.317 / 77.0 / 60.75`。
2. 这个结果可以称为**开放额外数据、同一重建评测器下的点估计 SOTA**，但不能简写成无条件“官方 SOTA”：
   - 没有官方 leaderboard 提交结果；
   - 原论文没有开放完整批量评测器，我们使用了可核查的重建评测器；
   - E004 相对此前最强单模型 SetPO 只提高 `+0.095` mIoU，95% CI 为 `[-0.706, +0.926]`，不显著；
   - 训练使用了额外合成 ESC-50 数据，不是同训练数据比较；
   - E003 的公开集聚合结果曾先被观察，因此 E004 不是完全纯净的首次 confirmatory test。
3. **最扎实的机制证据来自 RelTwin**。在三种子等预算实验中，SetPO 相对匹配的 RBEE parent 将 mIoU 从 `86.546` 提到 `89.904`，PairAcc@0.5 从 `81.458` 提到 `86.667`；两项 bootstrap CI 都严格大于 0。
4. **Clotho-Moment 不是第二个已确认 SOTA**。SetPO 全量 `6649` 条的 mIoU 为 `86.701`，确实高于 SpotSound 论文表中的 `85.6`，但同一评测脚本下的官方 SpotSound-A checkpoint 是 `86.854`，仍比 SetPO 高 `0.153`。准确表述应为：“超过论文公开表值，但未超过同评测器官方 checkpoint；R1@0.3 和 R1@0.5 有很小点估计提升，均未达到显著证据。”
5. **AEGBench 明确失败**：SetPO 的 Auto-AEG 核心 mIoU `38.934`，同评测器官方 checkpoint `40.820`，并低于公开强基线约 `48.0`。
6. **AudioGrounding v2 明确失败**：SetPO 全量 `997` 条 mIoU `69.610`，官方同评测器 `70.390`。
7. **SpanTool 是当前最新且故事性最强的方法方向，但不是当前 SOTA 方法**：它在内部 scale-cardinality dev 上得到可靠正提升（guided 版本 `+5.289` mIoU，95% CI `[+2.723,+8.116]`），却在公开 SpotSound 上原始自适应版本下降 `-1.054`，Clotho 只近似持平。其代码、单测和权重已保留，后续应把它当作“下一版主方法”，而不是现在论文表里的胜出结果。

### 0.2 当前建议的论文叙事

当前最有吸引力、也最不容易被评审说成纯粹拼模块的叙事是：

> **Generate → Intervene → Verify**：音频大模型先生成开放词汇的时间区间假设；模型随后在保持时间轴不变的条件下分别“保留候选区间”和“移除候选区间”，用充分性与必要性证据验证候选；最后通过在独立、低密度 LongNeedle 开发集上校准的保守选择器决定是否替换原预测。

RelTwin-RBEE/SetPO 是重要的训练机制支撑：它证明逆关系语义和集合边界可以被有针对性地学习。SpanTool 则是更完整的长期叙事：让 LLM 负责开放词汇语义，让非自回归结构化工具负责精确的可变基数时间集合解码。若 SpanTool 后续在两个公开 benchmark 上真正转正，它比 NOVA 更像一篇完整方法论文；在此之前，NOVA 是可用的 headline，SpanTool 是受控的下一阶段。

### 0.3 证据优先级

遇到聊天记录、旧口头结论与文件冲突时，按以下优先级判断：

1. 原始逐样本预测与锁定协议；
2. `results/claim_ledger.json`、`results/sota_scope.json`、`results/manifest.json`；
3. 实验目录中的 `summary.json`、`comparison.json`、`train_summary.json`；
4. 本文档和仓库 `README.md`；
5. 脱敏聊天记录中的过程性判断。

聊天中出现过“已经 SOTA”“Clotho 也是 SOTA”等简写，它们在当时是基于尚未完全统一评测器或只对论文表值的阶段性判断。本文档已按最终审计纠正。

## 1. 项目原始目标与硬约束

用户的原始需求并不是“从零发明一个宏大模型”，而是面向 ICASSP 的结果导向研究：

- 不做纯语音，希望结合音频大模型、LLM、CV/Agent 等热门技术；
- 优先寻找已有强 baseline 的薄弱点，在其上做小而有效的“改刀”；
- idea 只希望来自三类来源：
  1. 修补现有 baseline 的明确弱点；
  2. 把其他热门领域中成熟模块迁移到该 baseline；
  3. 提出该细分领域的新范式、训练方法或 loss；
- 目标优先级是可复现、能刷强结果、能写出清楚故事；极高 novelty 时可以容忍不是绝对 SOTA；
- 最初声明资源为 8×4090，实际首台服务器为 8×RTX 3090 且长期被占用；之后迁移到 2×32 GB vGPU 的 SeeTacloud 实例；
- 用户明确要求不要持续无目的调参，以免打乱论文叙事；后期实验必须服从已经收敛的核心问题。

这些约束决定了我们的研究方式：先把 benchmark、评测器和公开 checkpoint 审计清楚，再做“机制假设 → 独立开发门 → 公开集一次性应用”的证据链，而不是直接在 400 条 SpotSound 公开集上反复搜索参数。

## 2. 选题与 baseline 审计的演化

### 2.1 被放弃的第一版：VCT / AVSBench-Robust

最初曾考虑围绕音视听定位中的“Where Is Not Whether”问题，以 VCT 和 AVSBench-Robust 为入口。它看起来适合借用视觉 grounding/LLM 模块，但审计后存在几个硬问题：baseline 与 Robust 代码链较脆、数据和评测协议分散、复现成本高、对现有算力和投稿周期不友好。因此没有进入正式实现。

### 2.2 高 novelty 备选：关系反事实与 Audio Moment Retrieval

随后切换到音频时刻检索/开放词汇时间定位。第一轮方案曾把 QD-DETR、CASTELLA、DCASE 混在一起，但用户正确指出：这些 benchmark 并不自然支持同一叙事，QD-DETR 也不是这个任务上足够强的现代音频大模型 baseline。如果为了凑多个 benchmark 强行混合，评审会看到任务定义与评测协议不一致。

### 2.3 最终入口：SpotSound-A + SpotSound-Bench

重新审计后选择了 SpotSound：

- SpotSound-A 使用 Audio Flamingo 3 / Qwen2.5-7B 系列的音频语言模型和 LoRA，天然满足“音频 + 大模型”；
- SpotSound-Bench 是开放词汇、多区间的 audio temporal grounding，输出本质上是时间区间集合；
- 论文公开值已经较强，但存在三个明显薄弱点：
  1. 自然语言自回归输出并不擅长精确边界和可变基数集合；
  2. 对“先 A 后 B”和“先 B 后 A”这类逆关系缺少结构约束；
  3. 一个生成结果没有显式验证机制，长音频、低密度事件与困难样本存在候选选择空间；
- 官方公开了训练/推理代码、checkpoint 与 benchmark，但没有完整批量评测器，也没有完全复现原始 77.6k 训练集所需的全部语料和配置。

因此我们把 SpotSound-A 定为唯一 anchor baseline，所有训练和推理方法都从同一官方 checkpoint 出发，避免用弱 baseline 制造虚假提升。

## 3. Benchmark 与评测器审计

### 3.1 SpotSound-Bench 的真实规模

SpotSound-Bench v2 有 `400` 条 query annotation，但只有 `387` 个唯一音频文件；其中 `13` 条是复用音频上的额外 query，涉及 `11` 个多 query 文件。评测必须按 400 条 query 逐行进行，不能按文件名去重。相关事实固化在：

- `results/benchmark_identity_audit.json`
- `results/anchor_manifest.json`
- `results/code_provenance.json`

### 3.2 我们的 SpotSound 重建评测器

官方 SpotSound 仓库提供单样本推理、训练与输出格式，但没有可直接运行的全量 batch evaluator。我们的评测器遵循其自然语言响应格式，把预测解析为若干 `[start,end]`，进行合法化、裁剪和区间合并，然后计算预测时间集合与 GT 时间集合的时间并交比：

```text
SetIoU(P, G) = duration(union(P) ∩ union(G))
               --------------------------------
               duration(union(P) ∪ union(G))
```

R1@τ 按每条 query 的 SetIoU 是否不小于阈值 `τ` 统计。这个评测器对官方 checkpoint 得到 `58.317` mIoU，与论文 `57.9` 相差 `+0.417` 点，说明语义大体一致，但不能把二者当作完全同一个官方实现。

### 3.3 Auto-AEG 评测器为什么不能替换 SpotSound 主评测器

我们完整核对了 Auto-AEG 开放的评测代码。它与 SpotSound 不是“同一个指标的另一份代码”，而是至少有三处定义差异：

- 输出解析：Auto-AEG 期待 `<answer>[[...]]</answer>`，SpotSound 输出是自然语言中的时间区间；
- 区间聚合：SpotSound 以整个时间集合的并集做 Set-IoU；Auto-AEG 的核心 mIoU 更接近 GT-centric best match，具有 recall 倾向；
- 事件 F1：Auto-AEG 公共代码中的 `ev_F1` 是 onset tolerance 风格，而论文文字还描述过 Hungarian matching + IoU 的版本。

结论：

- 在 SpotSound-Bench 上，主表必须使用我们已经校验过的 SpotSound 重建评测器；
- Auto-AEG 风格分数只能做辅助分析；
- 在 AEGBench 上才使用其官方开放评测器；
- 不允许把两种 mIoU 混在同一列比较。

## 4. 第一条有效方法线：RelTwin-RBEE

### 4.1 问题假设

SpotSound-A 对事件词本身可能识别正确，却容易混淆逆关系。例如同一段音频中 A 和 B 都存在，query 为 “A before B” 与 “B before A” 时，模型可能返回相近甚至相同区间。这说明语义识别和关系顺序没有被充分解耦。

### 4.2 RelTwin 数据构造

RelTwin 使用 ESC-50 事件构造孪生关系样本：

1. 选取两类声音 A、B；
2. 生成 A→B 与 B→A 两个时间次序相反的混合；
3. 对同一音频构造互逆 query；
4. 保持事件集合、总时长与背景尽量一致，只改变次序和关系词；
5. 训练类与测试类 source-disjoint，十个 ESC-50 类留作测试；
6. 测试集为 40 对、每对 2 次重复，共 320 条逆关系 query。

这不是为了替代自然 benchmark，而是建立一个“如果方法真的学会关系，它必须在这里通过”的可证伪开发门。

### 4.3 RBEE：Relation-Boundary Exchange Equivariance

RBEE 的核心是同组 inverse-query 约束。设同一事件对的两个关系 query 为 `q_ab` 与 `q_ba`，模型给出边界/候选分布 `p_ab` 与 `p_ba`。交换关系后，正确分布应随对应关系发生可预测交换，而不是保持不变。训练同时包含：

- query-consistent 的时间边界监督；
- inverse relation 间的 exchange-equivariance；
- 普通定位样本 rehearsal，避免只会合成关系题而遗忘一般 query。

单次最早的干净对照已经显示：

| 方法 | PairAcc@0.5 | SwapError | mIoU | R1@0.5 |
|---|---:|---:|---:|---:|
| 普通 SFT | 53.13 | 40.00 | 74.28 | 75.31 |
| RBEE | **83.13** | **9.38** | **87.72** | **90.94** |

RBEE 的意义不是简单多训练一些数据，而是对“关系词变化时输出集合应如何变化”给出了明确结构约束。

## 5. SetPO：集合候选上的偏好优化

### 5.1 为什么 token-level SFT 不够

目标指标比较的是区间集合，但语言模型训练在 token 空间。两个文本输出可能 token 差异很大而时间集合等价；也可能文本看似接近却因为边界偏差导致 Set-IoU 很差。SetPO 把优化对象从单一字符串提升为结构化候选集合。

### 5.2 候选与目标分布

每个样本生成 6 个确定性 interval-set candidate。候选覆盖边界微调、集合收缩/扩张、关系交换等可解释变体。对候选 `c_i` 构造质量分数：

```text
quality(c_i) = α · SetIoU(c_i, y)
             + β · SoftIntervalF1(c_i, y)
```

再把候选质量转成目标偏好分布，与模型在这些候选上的归一化概率做 listwise 对齐。对 inverse relation pair 增加 Jensen-Shannon exchange 项，并保留 ordinary rehearsal。完整实现见：

- `scripts/setpo_candidates.py`
- `scripts/setpo_objective.py`
- `scripts/train_reltwin_setpo.py`

### 5.3 E001 为什么无效

E001 的三种子结果一度看起来可以聚合，但审计发现 parent training budget 不一致：seed 0 的 RBEE parent 是完整 256 step / 512 group，而 seed 1、2 来自更早的 64 step / 32 group。这样无法判断差异来自 SetPO 还是 parent 强弱。处理方式是：

- 不删除 E001；
- 把它作为失败稳定性审计保留；
- 原协议不改写，另存 post-hoc amendment；
- 重新运行等预算 E002。

文件：`experiments/protocols/E001_public_multiseed.json` 与 `experiments/protocols/E001_posthoc_amendment.json`。

### 5.4 E002 等预算三种子结果

E002 的每个 seed 都执行 RBEE 256 steps，再执行 SetPO 64 steps；公开 SpotSound 评测之前，必须先通过独立 RelTwin gate。

RelTwin 三种子均值：

| 方法 | mIoU | PairAcc@0.5 |
|---|---:|---:|
| RBEE-256 | 86.546 | 81.458 |
| RBEE-256 → SetPO-64 | **89.904** | **86.667** |
| 增量 | **+3.358** | **+5.208** |
| 95% CI | `[+2.469,+4.275]` | `[+2.917,+7.708]` |

SpotSound-Bench 同评测器结果：

| 方法 | mIoU | R1@0.3 | R1@0.5 |
|---|---:|---:|---:|
| 官方 SpotSound-A checkpoint | 58.317 | 77.000 | 60.750 |
| 普通 SFT seed 0 | 58.313 | 76.500 | 61.750 |
| RBEE 三种子均值 | 59.166 | 76.917 | 62.417 |
| SetPO 三种子均值 | **59.193** | 76.583 | **62.417** |

关键解释：

- SetPO 相对官方 checkpoint 的 mIoU 是 `+0.875` 点，三种子都超过官方；
- 层级 seed-record bootstrap CI 为 `[-0.243,+2.043]`，`p_nonpositive=0.063`，不满足预注册显著性门；
- SetPO 相对 matched RBEE 仅 `+0.027` mIoU，CI `[-0.313,+0.412]`；
- 因此公开 benchmark 的主要增益应归因于 RBEE，不能说 SetPO 本身在公开集得到稳定显著提升；
- ordinary SFT 的 mIoU 近似不变，说明不能把增益简单解释成“多训几步”。

原始证据集中在 `results/e002/`，决策文件是 `results/e002/sota_decision.json`。

## 6. Oracle 诊断：为什么需要候选验证

我们没有把 Oracle 当成方法，而是用它回答“已有 checkpoint 之间还剩多少可选择空间”。Oracle 使用 GT，在每条 benchmark query 上从候选池选 IoU 最高者；它不可部署、不能进入 SOTA 表，却能定位下一步的瓶颈。

| Oracle 候选池 | mIoU | R1@0.3 | R1@0.5 |
|---|---:|---:|---:|
| 3 个 RBEE seed | 59.924 | 77.75 | 63.25 |
| 3 个 SetPO seed | 60.073 | 78.00 | 63.50 |
| 6 个适配模型 | 60.466 | 78.50 | 63.50 |
| 官方 + SFT + 6 个适配模型 | **62.462** | **81.25** | **65.75** |

完整候选池比当时 SetPO headline 高 `3.269` mIoU，400 条里有 `199` 条可改善。提升主要集中在：

- 音频时长大于 60 秒：`+7.752`；
- 官方预测 IoU 小于 0.3 的困难样本：`+7.742`。

这说明继续把同一个 checkpoint 多训几十步不是最高价值方向；更合理的问题是：**在没有 GT 时，能否判断一个候选区间集合是否真的包含 query 证据？** 这直接催生 NOVA。

## 7. NOVA：Generate → Intervene → Verify

### 7.1 方法定义

NOVA 全称 Necessity-Oriented Verification over Audio。它把一个预测区间集合当作可证伪假设，而不是最终答案。对候选 `C` 构造两种音频干预：

1. `keep(C)`：只保留候选时间区域，其余区域替换为背景；若 query 证据确实位于 C，检测分数应保持较高——充分性；
2. `drop(C)`：只替换候选时间区域，其余时间轴不变；若 C 覆盖了必要证据，检测分数应明显下降——必要性。

冻结 SpotSound detector，以原音频、keep 音频、drop 音频上的 log-odds、区间几何、候选间一致性等特征描述每个候选。选择器只在独立合成开发集上拟合，公开 SpotSound 的 GT 不参与系数、标准化、阈值或逐行选择。

需要克制的措辞：这是**基于反事实式音频干预的候选验证**，不是严格意义上已经识别因果效应的 causal inference。干预后的分布可能偏离自然音频，因此必须用独立 gate 检验。

### 7.2 E003：高密度普通两事件集的失败

E003 在内部普通高密度两事件开发集上表现很好：NOVA `82.65`，最佳单候选 `79.82`，Oracle `82.79`。但迁移到 SpotSound 公开集后发生 ranking reversal：

| E003 公开版本 | SpotSound mIoU |
|---|---:|
| SetPO fallback | 59.234 |
| Drop-only | 58.869 |
| Full | 58.440 |
| Geometry | 58.427 |

失败原因不是代码跑错，而是开发分布的事件密度和候选排序与真实 SpotSound 不一致。这个结果非常重要：它否定了“只要内部 oracle 捕获率高，router 就会泛化”的想法，也促使我们采用低密度、长音频、更接近困难区域的 LongNeedle。

### 7.3 E004 LongNeedle 独立开发集

LongNeedle 由 160 个 60 秒 ESC-50 mixture 构成：

- 单次目标持续 1.2–2.5 秒；
- 目标出现 1/2/3 次的概率为 0.6/0.3/0.1；
- 平均 target density 为 4.76%；
- 排除 RelTwin 已使用 source clip；
- 固定划分 train 76、calibration 41、held-out 43；
- SetPO seed 1 是保守 fallback。

held-out 结果：

| 系统 | mIoU |
|---|---:|
| 最佳单一候选 | 23.334 |
| NOVA router | **24.358** |
| Oracle | 25.097 |

Router 捕获 `74.9%` 的候选 Oracle headroom，超过预设 20% gate，才允许一次性应用到公开 SpotSound。

### 7.4 E004 公开结果与边界

| 方法 | mIoU | R1@0.3 | R1@0.5 | R1@0.7 |
|---|---:|---:|---:|---:|
| SpotSound-A 论文表 | 57.900 | 76.50 | 59.50 | — |
| 官方 checkpoint，同评测器 | 58.317 | 77.00 | 60.75 | — |
| SetPO seed 1 fallback | 59.234 | 77.25 | 62.25 | 43.25 |
| **NOVA E004 locked full** | **59.329** | **78.75** | **62.25** | 42.75 |

NOVA 选择来源为官方 115 条、SFT 29 条、SetPO 256 条；相对 SetPO 有 74 win、55 loss、271 tie。它提升 mIoU `+0.095`、R1@0.3 `+1.5`，R1@0.5 不变，R1@0.7 `-0.5`。配对 CI 跨零，所以只声明点估计。

在看到 locked full 结果后做的 exploratory ablation 中，Keep-only 达 `59.515`、带七个 switch 的 geometry guard 达 `59.536`。它们不能取代预先锁定 headline，因为它们是在公开结果之后选择出来的；可以作为下一轮独立验证的候选。

E004 证据位于 `results/e004_longneedle/`。核心代码：

- `scripts/build_esc50_longneedle.py`
- `scripts/extract_nova_features.py`
- `scripts/fit_nova_router.py`
- `scripts/apply_nova_router.py`
- `scripts/nova.py`

## 8. 自然背景替换：必要的鲁棒性审计，但没有刷新 headline

最初 keep/drop 直接把被替换部分置零，这会制造不自然静音。我们实现了三种背景：

- `zero`：置零；
- `neighbor`：从相邻上下文复制/拼接背景；
- `matched_noise`：估计背景能量与频谱统计后生成匹配噪声；
- 边界使用 crossfade，尽量减小点击声和突变。

LongNeedle held-out：

| 背景方式 | router mIoU | best single | Oracle 捕获率 | gate |
|---|---:|---:|---:|---|
| matched-noise | 24.195 | 23.334 | 69.36% | 通过 |
| neighbor | 23.424 | 23.334 | 43.16% | 通过 |

但 matched-noise 应用到 SpotSound 后 mIoU 为 `59.160`，低于锁定 NOVA 的 `59.329`。结论是：自然替换增强了干预合理性和 robustness story，却没有成为新最强结果。实现保留在提交 `e5e9456` 及 `scripts/nova.py`，论文中适合作为消融/稳健性分析，而非 headline。

## 9. 第二 benchmark 审计

### 9.1 为什么不是 CASTELLA、DCASE 或 UnAV-100

- CASTELLA / DCASE 与当前“开放词汇、多区间 temporal grounding”的任务定义和输出形式不完全统一，为了凑表引入会使故事分裂；
- UnAV-100 有潜在价值，但数据获取、预处理和既有 protocol 对齐成本较高，当时不适合快速建立公平对照；
- 最终优先选择 Clotho-Moment 和 AEGBench，并补充 AudioGrounding v2 作为迁移压力测试。

### 9.2 Clotho-Moment 全量 6649 条

| 系统 | mIoU | R1@0.3 | R1@0.5 |
|---|---:|---:|---:|
| SpotSound 论文公开最好值 | 85.600 | 93.600 | 91.200 |
| 官方 SpotSound-A checkpoint，同脚本 | **86.854** | 93.488 | 90.856 |
| SetPO seed 1，同脚本 | 86.701 | **93.653** | **91.096** |
| SetPO − 官方同脚本 | -0.153 | +0.165 | +0.241 |

R1@0.5 差值 CI 为 `[-0.090,+0.572]` 点；mIoU 差值 CI 为 `[-0.380,+0.069]` 点。这里的正确结论是：SetPO 没有超过同评测器官方 checkpoint 的主指标，因此不是已证实的 Clotho 新 SOTA；它只在两个 recall 阈值上有微小点估计改善。

### 9.3 AEGBench 全量 9924 条

| 系统 | Auto-AEG core mIoU | SpotSound protocol mIoU |
|---|---:|---:|
| 官方 SpotSound-A checkpoint | **40.820** | **44.803** |
| SetPO seed 1 | 38.934 | 43.414 |

公开强结果约为 `48.0`，所以 AEGBench 不是当前有利 benchmark。还要注意其输出需要 canonical conversion；不能把临时转换后的数值包装成严格 leaderboard 提交。

### 9.4 AudioGrounding v2 全量 997 条

| 系统 | mIoU | R1@0.3 | R1@0.5 |
|---|---:|---:|---:|
| 官方 checkpoint | **70.390** | **88.867** | **76.229** |
| SetPO | 69.610 | 88.465 | 75.527 |

开发集上自然背景 NOVA 一度得到正信号，但 held-out 的保守 router 仍是 `82.317` 对官方 `82.592`，未通过。AudioGrounding 因此被明确拒绝为第二主 benchmark。

### 9.5 “先看 1000 条”的正确做法

用户提出先看 1000 条，有正信号再跑全量。第一次直接取 prefix-1000 后发现样本顺序有明显偏差，结果过于容易且不能代表全量。后来修正为：

1. 按稳定 hash/来源做 stratified 1000；
2. 在 manifest 中保存 `source_index`；
3. proposal、GT 和预测都用 `source_index` 对齐；
4. 只有 gate 主指标转正，才恢复全量；
5. 1000 条只用于节省算力，不能取代最后的全量报告。

## 10. SC-PSetPO：跨尺度训练尝试与负结果

### 10.1 动机

SetPO 在 RelTwin 强、SpotSound 略正、AEGBench/AudioGrounding 负，提示模型对事件持续时间、数量和数据域敏感。SC-PSetPO 试图加入 scale-cardinality balanced replay，并用多个目标轴而非单一 IoU 选择模型。

训练数据包含 432 条平衡 ESC-50 + Clotho 构造样本、996 条 rehearsal；训练 128 updates，学习率 `2e-6`，耗时约 3450 秒，峰值约 20.8 GiB。

### 10.2 结果

| 方法 | SpotSound mIoU | R1@0.3 | R1@0.5 | Clotho stratified-1000 mIoU |
|---|---:|---:|---:|---:|
| baseline | — | — | — | 89.783 |
| SC-PSetPO | 59.250 | 77.25 | 61.25 | 89.741 |
| equal-budget scalar control | 59.374 | 77.25 | 61.00 | 89.726 |

两者都没有得到跨 benchmark 正提升，也没有超过 NOVA headline。

### 10.3 审计发现

- 名为 Pareto 的目标仍然是固定 `0.5 × dominance + 0.5 × mean axes` 的 scalarization，不是真正的 Pareto-front optimization；
- 128 updates 只从 996 条中有放回抽样，较小 strata 的覆盖不均；
- 候选仍由 GT 规则生成，训练时强但推理仍依赖 AR 文本输出；
- 最根本矛盾没有解决：训练优化 interval set，推理却继续让 LLM 逐 token 写时间数字。

这轮负结果直接推动了 SpanTool：如果主误差来自输出参数化，就应改变输出范式，而不是继续在同一 AR 解码器上叠 loss。

## 11. SpanTool：完整版结构化输出方案

### 11.1 核心范式

SpanTool 将任务拆成两个互补角色：

```text
Audio + query
      │
      ▼
冻结 Audio Flamingo 3：开放词汇语义、query-aware 音频 hidden states
      │
      ▼
SpanTool：非自回归、可变基数、非重叠时间区间集合解码
      │
      ├── standalone：完全由结构头生成
      ├── refine：保留 AR proposal 基数，只校正边界
      └── guided：把 AR proposal 作为软先验，结构化 DP 重新解码
```

这个方向不是简单给 LLM 后面接一个 MLP。它改变了最终输出空间：从脆弱的时间数字 token 序列，改成受全局非重叠约束的 interval set。

### 11.2 一个决定性实现发现：必须 query-first

官方 SpotSound prompt 把 audio token 放在 query 之前。对 causal language model 而言，较早的 audio token hidden state 无法看到后面的 query；直接抽这些 state 训练定位头，得到的其实不是 query-aware 表征。

SpanTool runtime 改为：

```text
Instruction + Query + "Audio:" + audio tokens
```

于是 audio-token state 在因果注意力中能够访问 query。这个细节是 SpanTool 能否成立的前提，也是后续写论文时值得强调的实现洞见。见 `scripts/spantool_runtime.py::prepare_spantool_input`。

### 11.3 表征与多层融合

最终版本抽取 Audio Flamingo 3 的第 `-9` 与第 `-1` 层 audio-token hidden states。每层先独立 LayerNorm，再以可学习 softmax 权重融合。最佳 head 学得权重约：

```text
layer -9: 0.517962
layer -1: 0.482038
```

80-update pilot 中，只用最后层的 SC mIoU 为 `26.078`；双层融合为 `30.891`，提高约 `4.81` 点，说明中层时间线索有实际价值。

### 11.4 SpanToolHead 结构

最佳配置：

- 输入维度 3584；
- 两层融合；
- hidden dim 256；
- 2 层 TransformerEncoder；
- 8 个 attention head；
- dropout 0.1；
- primary pool = 5；
- coarse pool = 5；
- 16 维连续正弦时间特征，周期覆盖约 0.08–120 秒；
- 最大事件数 8；
- 头部可训练参数 `2,594,323`，基础模型与 SetPO adapter 冻结。

数据流：

1. 末尾不足一个 pool block 的 audio token 仍保留，不能静默丢失；
2. fine audio state 加入连续时间编码；
3. primary scale 经平均池化后输入时序 Transformer；
4. 再池化得到 coarse scale，并上采样回 primary 形成残差；
5. 输出 primary occupancy/onset/offset、fine onset/offset、coarse occupancy、count `0..8`、duration score 与 segment bias。

### 11.5 区间打分与动态规划

对于离散 start `s`、end `e`，区间分数由以下信息组成：

```text
S(s,e) = onset(s)
       + offset(e)
       + mean_occupancy(s:e)
       + duration_mlp(length, normalized_length)
       + segment_bias
```

所有 `e < s`、超过可用帧或违反限制的区间被 mask。给定事件数 `k` 后，使用 segmental forward DP 对所有包含恰好 k 个互不重叠区间的集合做 log-sum-exp，目标集合的 conditional segmental NLL 为：

```text
L_struct = log Σ_{Y: |Y|=k, non-overlap} exp S(Y) - S(Y_gt)
```

推理使用 Viterbi DP 找到恰好 k 个非重叠区间的最优集合。count 由 count head 给出；之后在 fine onset/offset logits 上做局部边界 refinement。相邻区间的搜索域以中点分隔，防止 refinement 后交叉。

### 11.6 完整损失

标准训练权重为：

```text
L = 1.00 * L_structural
  + 0.50 * L_occupancy
  + 0.25 * L_boundary
  + 0.25 * L_count
  + 0.10 * L_scale_consistency
  + 0.00 * L_risk
```

- occupancy 使用正负平衡 BCE；
- boundary 同时监督 primary 与 fine onset/offset；
- count 是 0–8 类交叉熵；
- scale consistency 约束 coarse occupancy 与池化后的 primary probability；
- 代码包含 perturb-and-MAP structured risk：用 Gumbel 扰动生成多个结构化集合，并按 soft F1、Set-IoU、count score 的质量风险训练；实测 risk run 没有改善，因此默认权重为 0，不作为当前主贡献。

### 11.7 训练数据与资源

最佳标准 head 从 E002 seed-1 SetPO adapter 与早期融合 head warm start，训练 1000 updates，gradient accumulation 4，head LR `2e-4`，warmup 50，cosine schedule。各 source：

| source | manifest 行数 | 抽样权重 | 4000 microstep 实际抽样数 |
|---|---:|---:|---:|
| AudioGrounding train | 8020 | 0.55 | 2157 |
| RelTwin rehearsal | 440 | 0.10 | 382 |
| scale-cardinality | 374 | 0.15 | 632 |
| Clotho train subset | 327 | 0.20 | 829 |

验证集：SC 58 条，AR baseline mIoU `35.469`；Clotho 59 条，AR baseline `89.996`。训练耗时 `1761.9` 秒，峰值 `17128.8` MiB，单张 32 GB vGPU 足够。

### 11.8 训练曲线揭示的冲突

| update | SC mIoU | SC Δ | Clotho mIoU | Clotho Δ |
|---:|---:|---:|---:|---:|
| 250 | 27.386 | -8.083 | 64.718 | -25.278 |
| 500 | 41.272 | +5.802 | 63.415 | -26.581 |
| 750 | **42.283** | **+6.814** | 64.464 | -25.533 |
| 1000 | 40.051 | +4.581 | **66.080** | -23.917 |

独立 standalone head 对 Clotho 的单事件、精确边界分布存在严重遗忘/失配。因为 selection score 取跨验证源最差 improvement，最佳仍是 update 1000，但它没有真正解决跨域冲突。

### 11.9 AR proposal guided/refine

为利用 AR checkpoint 已有的强单事件能力，引入两种组合：

- `refine`：固定 proposal cardinality，在 ±0.8 秒等局部窗口用 fine boundary logits 吸附边界；
- `guided`：把 proposal 转成 start/end/occupancy soft prior，加到结构化 segment score 后再做 DP；proposal cardinality 作为 count。

默认 guided 在开发集：

| 数据 | AR baseline | guided | ΔmIoU | 95% CI | win/tie/loss | Oracle |
|---|---:|---:|---:|---|---|---:|
| SC dev 58 | 35.469 | **40.759** | **+5.289** | `[+2.723,+8.116]` | 40/8/10 | 41.326 |
| Clotho dev 59 | **89.996** | 89.530 | -0.466 | `[-0.788,-0.164]` | 12/2/45 | 90.190 |

更高 proposal anchor 或更窄 radius 能保住部分 SC 增益，但 Clotho 仍为负。直接 refine 是 SC `+2.267`、Clotho `-0.303`，也没有同时转正。

### 11.10 公开集与 router 审计

原始 adaptive 公开结果：

| 数据 | baseline mIoU | SpanTool candidate | Δ | 95% CI |
|---|---:|---:|---:|---|
| SpotSound 400 | 59.234 | 58.180 | **-1.054** | `[-1.753,-0.393]` |
| Clotho stratified 1000 | 85.638019 | 85.638843 | +0.000824 | `[-0.01254,+0.01434]` |

SpotSound 上 AR+SpanTool 的逐样本 Oracle 为 `60.210`，比 baseline 高 `0.976`，说明候选有互补性；但简单 agreement gate、cardinality gate、residual blend 都无法稳定选择。

随后训练 14 维 confidence router，数据为 SC 374 + Clotho 327 共 701 条。交叉验证：SC `+1.423`、Clotho `+0.028`；独立 dev：SC `+1.558`、Clotho `+0.017`。可是一到公开 SpotSound 就下降 `-1.010`，Clotho 仍只有 `+0.000824`。这再次表明 source-domain router 不能被当作免费泛化器。

为了维护清楚叙事，router 与 adaptive residual 的临时代码已经从最终 Git 分支移除，没有推送；负结果 JSON 保留在交接包。仓库中保留的是 SpanTool 核心结构、训练、三种解码方式、比较脚本和测试。

### 11.11 修复过的关键 bug

1. **NaN gradient**：区间长度在 mask 前进入 `log1p`，无效负长度已经产生 NaN；修复为先 clamp 再算特征。
2. **Viterbi 速度**：最初 Python 三重循环约 11 分钟/样本；改为向量化/NumPy DP 后约 1.36 秒/样本。
3. **尾部 token 丢失**：池化必须保留 partial final block。
4. **query 不可见**：改成 query-first prompt。
5. **边界交叉**：fine refinement 的相邻区间搜索域使用中点分割。
6. **缓存污染**：evaluator cache key 加入代码 hash、checkpoint hash、decode mode、proposal hash 和参数。
7. **1000 条错位**：stratified manifest 保留 `source_index`，proposal 以原索引对齐。
8. **随机数耦合**：structured risk 使用独立 RNG，避免改变数据抽样序列。
9. **FLAC 时长异常**：AudioGrounding 中无法可靠读取的 FLAC 被显式修复/重物化。

SpanTool 最终提交顺序：

- `8bc1961`：完整 SpanTool 实现；
- `f42a6e6`：paired SpanTool oracle audit；
- `f2f7838`：评测不变量、缓存与索引审计。

当前代码测试为 `57 passed`。这只能证明实现内部一致，不能替代公开 benchmark 正结果。

## 12. 截至冻结日的完整结果矩阵

### 12.1 哪些结论可以写进摘要/主表

| 结论 | 状态 | 允许的表述 |
|---|---|---|
| NOVA SpotSound 59.329 | 可用 headline | open-extra-data, reconstructed-same-harness point-estimate SOTA；同时报告 CI 跨零 |
| RBEE/SetPO RelTwin 稳定提升 | 强机制证据 | 三种子等预算、CI 严格为正；证明逆关系与集合偏好训练有效 |
| SetPO SpotSound 59.193 mean | 可用支撑 | 三种子都超过官方同评测器；主要收益来自 RBEE，SetPO 对 RBEE 的公开增量很小 |
| 自然背景干预 | 可用 robustness ablation | LongNeedle gate 通过，但 SpotSound 未刷新 headline |
| Oracle 62.462 | 仅诊断 | 候选池潜在上界/错误分析，不是可部署结果 |

### 12.2 哪些结论不能写成 SOTA

| 结果 | 原因 |
|---|---|
| Clotho SetPO 86.701 | 低于同评测器官方 checkpoint 86.854；只能说高于论文表值 85.6 |
| AEGBench SetPO 38.934 | 明显低于官方与公开强方法 |
| AudioGrounding SetPO 69.610 | 低于同评测器官方 70.390 |
| SpanTool SC dev +5.289 | 内部小型合成/开发集正结果，公开 SpotSound 为负 |
| E004 exploratory Keep/Geometry 59.515/59.536 | 看过公开结果后选出的 post-hoc 分支，需新独立 test 才能升级 |
| Oracle 62.462 | 使用 GT 逐行选择，不可部署 |

### 12.3 “是不是纯缝模块”的最终回答

不是纯粹堆模块，但也不能夸成所有组件都是首次提出：

- preference learning、equivariance、rehearsal、occlusion、线性 router、dynamic programming 都是已知工具；
- 新意在于对 inverse audio relation 和 set-valued temporal grounding 的具体问题化：RelTwin paired construction、RBEE exchange constraint、SetPO interval-set listwise target；
- NOVA 的新意在于把时间区间候选变成 keep/drop 可证伪假设，并在独立 LongNeedle 上校准后保守选择；
- SpanTool 的新意更强：query-first 大模型时序状态 + exact conditional segmental set likelihood + 可变基数非重叠集合解码，且把 AR 输出降为可选 proposal prior。

最容易被评审攻击的是“方法像工程组合、公开提升小且不显著、额外合成数据、只有一个真正胜出的 benchmark”。写作时要用机制实验和失败边界回应，而不是隐藏这些问题。

## 13. Git 仓库结构与提交历史

### 13.1 关键目录

```text
spotsound_release/
├── README.md                         # 当前公开结果、协议和基本复现
├── DEVELOPMENT_HISTORY.md            # 本文：完整过程与交接
├── requirements.txt                  # 固定依赖范围
├── experiments/protocols/            # E001–E004 与 RBEE 锁定协议
├── scripts/
│   ├── spotsound.py                  # 模型加载、prompt、输出解析等共用逻辑
│   ├── interval_metrics.py           # SpotSound interval-set 指标
│   ├── build_esc50_reltwin.py        # RelTwin 构造
│   ├── train_reltwin_micro.py        # RBEE/SFT
│   ├── setpo_candidates.py           # 六候选集合
│   ├── setpo_objective.py            # SetPO 目标
│   ├── train_reltwin_setpo.py        # SetPO 训练
│   ├── build_esc50_longneedle.py     # LongNeedle
│   ├── nova.py                       # keep/drop 干预与自然背景
│   ├── extract_nova_features.py      # 冻结 verifier 特征
│   ├── fit_nova_router.py            # 独立开发集 router
│   ├── apply_nova_router.py          # 标签无关公开应用
│   ├── prepare_public_benchmarks.py  # Clotho/AEGBench 等标准化
│   ├── prepare_spantool_splits.py    # SpanTool train/dev 划分
│   ├── spantool.py                   # 结构头、DP、loss、decode
│   ├── spantool_runtime.py           # query-first 输入与 hidden state
│   ├── train_spantool.py             # 多源训练
│   └── evaluate_spantool.py          # standalone/refine/guided 评测
├── results/                          # 已提交的 SpotSound 原始证据
└── tests/                            # 57 个通过的测试
```

### 13.2 关键提交

| commit | 内容 |
|---|---|
| `1669afc` | 首个 RelTwin-RBEE / SetPO 证据发布 |
| `8d06c8d` | Oracle headroom 诊断 |
| `730f035` | NOVA LongNeedle 证据；当前 main 指向此提交 |
| `e5e9456` | 自然背景替换实现 |
| `8bc1961` | 完整 SpanTool 实现 |
| `f42a6e6` | SpanTool paired Oracle audit |
| `f2f7838` | evaluation invariants 与最终审计修复 |

冻结前远端：

```text
origin/main                 730f03556728027e0edb8572e0b48a96c43dc911
origin/codex/spantool-full  f2f7838a98a5e3bd19260547b60db0f3e79e4ffc
```

本文提交后 `codex/spantool-full` 会再前进一个文档 commit。不要把 main 自动视为最新代码；SpanTool 只在开发分支。

## 14. 服务器迁移、当前环境与资产

### 14.1 迁移历史

第一台 `zhengce@222.20.99.70` 是 8×RTX 3090 24 GB，但持续没有满足双卡空闲条件。自动监控后来关闭。项目随后运行在旧 SeeTacloud `connect.westb.seetacloud.com:30832`，再迁到当前实例 `connect.westb.seetacloud.com:42339`。

旧项目约 52 GiB：datasets 21G、models 16G、env 7.5G、experiments 3.4G、pip cache 4G。迁移时排除了未使用的约 18.4G FSD50K split archives 和可重建 cache；核心数据约 3.562G、环境、基础模型与 32 个 adapter 的聚合 hash 在源/目标一致。迁移后 smoke：GT `8.1–10.9s`，预测 `8.01–11.01s`，约 1.41 秒，峰值约 16.23 GiB。

### 14.2 当前服务器状态（2026-09-02 审计）

```text
host: autodl-container-e6984d9b33-a7743838
GPU: 2 × NVIDIA vGPU-32GB, 32760 MiB each
审计时 GPU: 两卡各 1 MiB / 0% utilization
磁盘: /dev/md0 350G total, 157G used, 194G available
Python: 3.10.21
PyTorch: 2.8.0+cu128
Transformers: 5.9.0
PEFT: 0.18.1
```

主要路径：

```text
/root/autodl-tmp/SpotSound-ICASSP/
├── spotsound_release/                  # 服务器工作副本
├── env/                                # Python 环境
├── models/                             # AF3 base + official adapter
├── datasets/                           # SpotSound/Clotho/AEGBench/AudioGrounding
└── autoresearch/06_experiments/
    ├── data/                            # RelTwin/LongNeedle/SC/SpanTool manifests
    └── runs/                            # 所有训练和评测产物
```

空间快照：repo 9.1M、models 16G、datasets 72G、SpanTool runs 109M。

### 14.3 一个重要的 Git 风险

服务器 `spotsound_release` 当时显示分支名 `codex/spantool-full`，但 HEAD 仍是 `e5e9456`，并存在大量由本地同步过去的 modified/untracked SpanTool 文件，包括后来已放弃的临时 router 文件。它是实验现场，不是干净 Git 真相源。

因此：

- 不要在服务器现场直接 `git reset --hard`；可能破坏仍需留证的未跟踪结果或脚本；
- 代码真相以 GitHub `codex/spantool-full` 为准；
- 实验产物真相以服务器 `runs/` 和交接包中的 metadata 为准；
- 如果要继续实验，建议在服务器另 clone 一个干净目录，或先归档现场再 checkout。

### 14.4 交接包内附的关键权重

GitHub 因许可和体积不包含模型权重。交接 ZIP 额外附：

1. E002 seed-1 SetPO LoRA adapter，约 78 MiB；
2. SpanTool full-fusion warm-start head，约 10 MiB；
3. SpanTool expanded-standard-v2 最佳 head，约 10 MiB；
4. 对应配置与 SHA-256。

校验值：

```text
SetPO adapter_model.safetensors
b2068cd13fd207f4641360ae4df87414eed2d6e626d5fb8b272e25ff04f6b800

SpanTool warm-start spantool_head.pt
db85bf4447979a9e0968f42fc9db93eaefeb2fa3fbf278fc77f6f5032e7ee1fb

SpanTool spantool_head.pt
29251c9b9176281ce099575f24c3659c5806166f7e7f9e959513af1c715aa2c8

SpanTool spantool_config.json
9d3a16151832e0df110ae6f684c5a0b97320487a39a09e4f9a33d10eee2cd8de
```

基础 Audio Flamingo 3、官方 SpotSound adapter 和 benchmark 音频不放入 ZIP，按固定公开 revision 下载。

## 15. 从零复现与继续实验

### 15.1 获取代码与环境

```bash
git clone --branch codex/spantool-full \
  https://github.com/tzcinhust/spotsound.git spotsound_release
cd spotsound_release

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

已测核心版本：Python 3.10.21、torch 2.8.0+cu128、transformers 5.9.0、peft 0.18.1。需求文件允许的是经过测试的窄版本范围，若出现数值漂移，先回到这些精确版本。

### 15.2 下载固定公开资产

```bash
hf download nvidia/audio-flamingo-3-hf \
  --revision 7d4bae64ee29878af6504ae6f6bb3e40492838ad \
  --local-dir models/audio-flamingo-3-hf

hf download Loie/SpotSound \
  --revision 07eca9f048599fde92a55f2376e429deaa73c21a \
  --local-dir models/SpotSound

hf download Loie/SpotSound-Bench --repo-type dataset \
  --revision 2e64ad197b90d6bd3b43efc73738cb9f86810735 \
  --local-dir datasets/SpotSound-Bench
```

上游固定项：SpotSound paper arXiv v2（2026-08-10）；官方代码 commit `d60c0e214ab685d0aa916db655fc59191900fb92`。Audio Flamingo 3 有非商业研究限制；上游代码、adapter、ESC-50 与 benchmark 音频各自保留原许可。本仓库 MIT 只覆盖本仓库原创代码。

### 15.3 构造 RelTwin

```bash
python scripts/build_esc50_reltwin.py \
  --metadata data/ESC-50/meta/esc50.csv \
  --audio-dir data/ESC-50/audio \
  --output-dir data/reltwin_esc50_v1/audio \
  --train-manifest data/reltwin_esc50_v1/train.json \
  --test-manifest data/reltwin_esc50_v1/test.json \
  --rehearsal-manifest data/reltwin_esc50_v1/rehearsal.json \
  --train-audios 256 --test-pairs 40 --test-replicates 2 --seed 0
```

### 15.4 复现 E002 三种子

```bash
PROJECT_ROOT="$PWD" \
BASE_MODEL=models/audio-flamingo-3-hf \
SPOTSOUND_ADAPTER=models/SpotSound \
RELTWIN_DATA=data/reltwin_esc50_v1 \
SPOTSOUND_ANNOTATIONS=datasets/SpotSound-Bench/annotations_processed.json \
SPOTSOUND_AUDIO_DIR=datasets/SpotSound-Bench/audio \
OUTPUT_DIR=outputs/e002_matched_pipeline \
bash scripts/run_e002_matched_pipeline.sh
```

脚本只有在每个 SetPO seed 都改善 matched RBEE 的 RelTwin mIoU 与 PairAcc@0.5，且三种子均值两项都提高后，才进入 SpotSound 公开评测。

聚合：

```bash
python scripts/summarize_public_multiseed.py \
  --baseline results/official_spotsound_a/public_predictions.jsonl \
  --method \
    outputs/e002_matched_pipeline/seed_0/setpo/public_predictions.jsonl \
    outputs/e002_matched_pipeline/seed_1/setpo/public_predictions.jsonl \
    outputs/e002_matched_pipeline/seed_2/setpo/public_predictions.jsonl \
  --output outputs/e002_matched_pipeline/public_multiseed.json \
  --samples 50000 --seed 20260831
```

### 15.5 构造 LongNeedle 与复现 NOVA gate

```bash
python scripts/build_esc50_longneedle.py \
  --metadata data/ESC-50/meta/esc50.csv \
  --audio-dir data/ESC-50/audio \
  --output-dir data/longneedle_esc50_v1/audio \
  --manifest data/longneedle_esc50_v1/test.json \
  --exclude-manifest data/reltwin_esc50_v1/train.json \
  --exclude-manifest data/reltwin_esc50_v1/test.json

PROJECT_ROOT="$PWD" \
BASE_MODEL=models/audio-flamingo-3-hf \
OFFICIAL_ADAPTER=models/SpotSound \
SFT_ADAPTER=outputs/reltwin_esc50_v1/sft/adapter \
SETPO_ADAPTER=outputs/e002/seed_1/setpo/adapter \
LONGNEEDLE_DATA=data/longneedle_esc50_v1 \
OUTPUT_DIR=outputs/e004_longneedle \
bash scripts/run_e004_longneedle_gate.sh
```

退出码 42 表示开发门失败，不应继续跑公开集。通过后再提取公开特征并使用锁定 router：

```bash
python scripts/extract_nova_features.py \
  --base models/audio-flamingo-3-hf \
  --verifier-adapter models/SpotSound \
  --manifest datasets/SpotSound-Bench/annotations_processed.json \
  --audio-dir datasets/SpotSound-Bench/audio \
  --candidate official=results/official_spotsound_a/public_predictions.jsonl \
  --candidate sft=results/controlled/sft_seed0/public_predictions.jsonl \
  --candidate setpo=results/e002/seed_1/setpo/public_predictions.jsonl \
  --output outputs/e004_public/features.jsonl

python scripts/apply_nova_router.py \
  --features outputs/e004_public/features.jsonl \
  --model outputs/e004_longneedle/router.json \
  --output outputs/e004_public/predictions.jsonl \
  --summary outputs/e004_public/summary.json
```

### 15.6 准备 SpanTool 数据

仓库提供：

```bash
python scripts/prepare_audiogrounding_train.py --help
python scripts/prepare_clotho_train_subset.py --help
python scripts/build_esc50_scale_cardinality.py --help
python scripts/prepare_spantool_splits.py --help
```

先按服务器 metadata 中的 manifest SHA-256 检查构造是否一致：

```text
audiogrounding_train.json  f81e30e2b85a0b452335ddd53f4b9fe0153a1e1994827887e7bbdec6e79e0977
sc_train.json              0416b8de9218e9eba2054b4e425ac549a670ac90722134109ef5ec31e9d25323
sc_dev.json                606c2058dc731a5ed38e6be58f505bf66b8c162069a382cf3267dbd1a3f33aa4
clotho_train.json          f914fed57d1f83f93563cfa13f3f8482b86a497cfc8abca6746e01c83f063b5b
clotho_dev.json            96fada7e7f7fd9953f2d68138862fa8f0bfcf3ef72891f1a20caf7cac22fea04
```

### 15.7 复现 SpanTool 最佳训练

将交接包的 SetPO adapter 与 warm-start head 解压到本地后：

```bash
python scripts/train_spantool.py \
  --base models/audio-flamingo-3-hf \
  --adapter weights/setpo_seed1_adapter \
  --initial-head weights/spantool_warmstart_head \
  --output-dir outputs/spantool_expanded_standard_v2 \
  --train-source audiogrounding data/spantool_v1/audiogrounding_train.json datasets/AudioGrounding-v2/repaired_audio 0.55 \
  --train-source rehearsal data/spantool_v1/rehearsal_train.json data/reltwin_esc50_v1 0.10 \
  --train-source sc data/spantool_v1/sc_train.json data/sc_rehearsal_v1 0.15 \
  --train-source clotho data/spantool_v1/clotho_train.json datasets/Clotho-Moment/train_subset/materialized_full 0.20 \
  --validation-source sc data/spantool_v1/sc_dev.json data/sc_rehearsal_v1 35.46929318075019 \
  --validation-source clotho data/spantool_v1/clotho_dev.json datasets/Clotho-Moment/train_subset/materialized_full 89.99626859188072 \
  --audio-layers -9 -1 \
  --hidden-dim 256 --transform-layers 2 --attention-heads 8 \
  --primary-pool 5 --coarse-pool 5 --max-events 8 \
  --updates 1000 --gradient-accumulation 4 \
  --head-learning-rate 2e-4 --weight-decay 0.01 --warmup-ratio 0.05 \
  --structural-weight 1.0 --occupancy-loss-weight 0.5 \
  --boundary-loss-weight 0.25 --count-loss-weight 0.25 \
  --consistency-loss-weight 0.1 --risk-weight 0.0 \
  --eval-every 250 --validation-limit 0 --seed 20260902
```

注意：最佳 run 是从 `full_fusion_standard_s20260902/best` warm start；只从随机 head 开始不会严格复现同一轨迹。交接包已经包含该 warm-start head。

### 15.8 SpanTool 评测

```bash
python scripts/evaluate_spantool.py \
  --base models/audio-flamingo-3-hf \
  --adapter weights/setpo_seed1_adapter \
  --spantool-checkpoint weights/spantool_best_head \
  --annotations data/spantool_v1/sc_dev.json \
  --audio-dir data/sc_rehearsal_v1 \
  --proposal-predictions outputs/sc_ar_predictions.jsonl \
  --decode-mode guided --proposal-weight 1.0 \
  --refinement-radius-seconds 0.8 \
  --predictions outputs/sc_guided_predictions.jsonl \
  --summary outputs/sc_guided_summary.json
```

同一命令可换 `--decode-mode standalone` 或 `refine`。评测缓存如果存在，也必须检查 summary 中 checkpoint/code/proposal hash 是否相同，不能仅凭输出文件名复用。

## 16. 失败实验清单：不要重复踩坑

| 尝试 | 看到的正信号 | 最终问题 | 处理 |
|---|---|---|---|
| E001 不等预算三种子 | 聚合值看似可用 | parent 训练预算不匹配 | 作废为 SOTA 证据，保留审计；用 E002 重跑 |
| E003 高密度 NOVA | 内部 82.65 接近 Oracle 82.79 | 公开集 ranking reversal，full 58.44 | 放弃该开发分布 |
| 自然 matched-noise NOVA | LongNeedle gate 69.36% 捕获率 | SpotSound 59.160，低于 59.329 | 只作 robustness ablation |
| AEGBench 迁移 | 可使用完整官方 evaluator | SetPO 38.934 < 官方 40.820 < 强方法约 48 | 不作有利第二 benchmark |
| AudioGrounding 迁移 | dev/局部自然 NOVA 有正趋势 | 全量 SetPO 与 held-out router 均落后 | 拒绝为第二主 benchmark |
| Clotho prefix-1000 | 数值高、运行快 | prefix 样本分布偏，不能代表全量 | 改为 stratified + source_index |
| SC-PSetPO | 试图统一尺度/基数 | SpotSound 与 Clotho 都未同时改善 | 不再沿 AR loss 继续堆叠 |
| SpanTool standalone | SC 中期可 +6.8 | Clotho 严重退化 | 引入 proposal prior，不直接替换 AR |
| SpanTool guided | SC +5.289 且 CI 正 | Clotho -0.466；SpotSound -1.054 | 仅保留机制结果，公开主表不使用 |
| SpanTool confidence router | source CV 与独立 dev 两域均正 | SpotSound public -1.010 | 临时代码移除，元数据留证 |
| SpanTool structured risk | 理论上更贴近集合指标 | 与标准 run 无改善/近似一致 | 默认关闭，不包装成贡献 |

这些失败共同说明：当前真正困难的不是生成更多候选，而是**跨数据域校准“何时相信结构化候选”**。任何新的 gate/router 必须在与 SpotSound 公开集无标签隔离的 target-like calibration 上验证，或者设计成理论上单调不伤害的更新；否则内部正信号不可信。

## 17. 下一阶段最值得做的实验

用户已明确不希望无目的乱调。建议只沿以下主线推进，每一步都设停止条件。

### 17.1 第一优先：把 SpanTool 变成“只修边界、不改语义集合”的残差工具

当前失败主要来自 SpanTool 改变了 AR 候选的区间数量或整体位置。下一版不要再训练普通逐区间 router，而应让模型直接预测受限 residual：

```text
Δstart_i, Δend_i ∈ [-r, r]
prediction_i = proposal_i + residual_i
```

约束：

- 默认严格保留 AR cardinality；
- 区间次序与非重叠通过投影/DP 保证；
- residual 初始为 0，因此初始模型精确等于 baseline；
- 训练 loss 直接优化 paired improvement，并对负 improvement 加不对称惩罚；
- 允许 abstain：置信不足时 residual 必须回到 0。

这比后验 router 更有叙事一致性：LLM 生成语义 proposal，SpanTool 只作为“边界校准工具”调用。它也天然适合 Clotho 的单区间高精度分布，不会因为重预测 cardinality 而大幅伤害。

停止条件：SC dev 与 Clotho dev 都必须 mIoU > baseline，且各自 paired bootstrap 下界不低于预设容忍线；否则不跑公开 SpotSound。

### 17.2 第二优先：训练时加入 target-like 长度/密度重加权

SC 数据与 SpotSound/Clotho 的长度、target density、事件数分布不同。不要用公开 GT 调参；可从不含标签的目标音频长度统计和 query 文本统计得到 covariate，再对 AudioGrounding/SC/Clotho 训练样本做 density-ratio reweighting。重点不是扩大数据，而是减少 validation-source conflict。

应报告：每个 source 的长度、事件数、coverage 分布；训练抽样前后 KL/EMD；按长度/基数分层的 ΔmIoU。若只提高合成 SC 而目标样式 dev 不动，应立即停止。

### 17.3 第三优先：建立新的未触碰 confirmatory split

E004 的最大证据弱点是 E003 已看过 SpotSound aggregate。若论文要更强，需准备一个未用于任何设计的 test：

- 优先寻找 SpotSound 官方隐藏/后续 test 或联系作者提交；
- 若不可得，可在另一个任务定义严格一致的公开 benchmark 上预注册协议；
- 先冻结代码、checkpoint hash、gate、所有阈值，再一次性评测；
- 不以同一 test 上探索出的 Keep-only/Geometry 59.515/59.536 作为 confirmatory 结果。

### 17.4 不建议继续做的事情

- 在 SpotSound 400 条上继续扫描 gate 阈值；
- 把更多通用 LLM/Agent 模块堆进 pipeline，只为增加名词；
- 再找任务定义不一致的 benchmark 凑数量；
- 用 Oracle 逐行选择后的值作为可部署上限之外的主结果；
- 只报告比论文表高、却不报告同评测器官方 checkpoint；
- 忽略 CI、额外数据和 public reuse 的限定。

## 18. 推荐论文结构

### 18.1 如果近期必须投稿：NOVA 主线

建议标题方向：**Generate, Intervene, Verify: Counterfactual Candidate Auditing for Open-Vocabulary Audio Temporal Grounding**。

文章结构：

1. SpotSound-A 能生成开放词汇区间，但生成不是验证；
2. RelTwin 揭示 inverse relation 与边界集合脆弱性；RBEE/SetPO 提供更强候选；
3. Oracle 显示候选互补性集中在长音频和困难样本；
4. NOVA 用 keep/drop 干预检验充分性/必要性；
5. LongNeedle 提供公开标签隔离的低密度 calibration；
6. SpotSound 得到 59.329 point estimate，并诚实报告统计和自然背景消融；
7. Clotho/AEGBench/AudioGrounding 作为泛化边界，而不是硬说全部提升。

这条线的风险是公开提升小、统计不显著、第二 benchmark 未赢。写作上要把贡献重心放在新评测范式、机制诊断与透明证据，而不是把 `+0.095` 写成巨大性能突破。

### 18.2 如果允许再做一轮核心实验：SpanTool 主线

建议标题方向：**Language Models Propose, Structured Tools Localize: Exact Set Decoding for Audio Temporal Grounding**。

必须满足后才切换 headline：

- SpotSound 至少超过锁定 NOVA 59.329，且 R1 指标不能全面退化；
- Clotho 至少超过同评测器 official 86.854，而不是只超过论文表 85.6；
- 至少一个 target-like 未触碰开发/测试集给出一致正提升；
- residual/abstention 设计固定在公开评测之前；
- 提供 standalone、refine、guided/residual、query-first、layer fusion、structured NLL 消融。

### 18.3 推荐的 claim 句式

可写：

> Under a reconstructed evaluator that reproduces the released SpotSound-A checkpoint within 0.42 mIoU points of the paper, NOVA obtains a new open-extra-data point estimate of 59.33 mIoU on the 400-query SpotSound-Bench.

必须紧接：

> The gain over the strongest pre-registered single candidate is 0.095 points and is not statistically significant; we therefore claim a point estimate rather than a definitive leaderboard improvement.

RelTwin 可写：

> Across three matched-budget seeds, SetPO improves RelTwin mIoU by 3.36 points and paired inverse-relation accuracy by 5.21 points, with both bootstrap intervals strictly above zero.

不可写：

- “NOVA significantly outperforms SOTA”；
- “SOTA on SpotSound and Clotho”；
- “official leaderboard SOTA”；
- “no extra data”；
- “causal effect is identified”；
- “SpanTool improves public benchmarks”。

## 19. 队友接手清单

### 19.1 第一天先做

1. Clone `codex/spantool-full`，核对当前 commit 与本文末尾 manifest；
2. 阅读 `README.md`、本文和 `results/claim_ledger.json`；
3. 解压交接 ZIP，校验 `ARTIFACT_MANIFEST.sha256`；
4. 将三组关键权重放到 `weights/`；
5. 下载固定 revision 的 AF3、官方 adapter、SpotSound-Bench；
6. 运行 `python -m pytest -q`，预期 57 passed；
7. 用提交的 E004 预测重算一次指标，确认 evaluator 一致；
8. 再做一个 1–2 条样本的 SpanTool guided smoke，不要立即全量训练。

### 19.2 继续跑实验前

- 给每个新实验写简短 protocol JSON：假设、数据、主指标、gate、失败停止条件；
- 保存 Git commit、dirty diff hash、manifest hash、checkpoint hash、环境版本；
- 公开集只在 gate 通过后运行；
- 新方法必须与同一 official/SetPO proposal、同一 evaluator、同一 400/6649 行比较；
- 首 1000 条必须 stratified，不能 prefix；
- 所有 candidate merge 必须按 `source_index`；
- 把失败结果保留，不能覆盖成最新目录。

### 19.3 服务器现场注意

- 登录凭据不写入仓库或交接文档；由项目负责人单独分发；
- 当前实例入口为 `connect.westb.seetacloud.com:42339`，用户名 root；
- 运行前先 `nvidia-smi`，不要假设两张卡一直空闲；
- SpanTool 单卡峰值约 17.1 GiB；AF3 单模型推理约 16.2 GiB；双卡可以各跑一个独立 shard；
- 服务器工作树是 dirty experiment snapshot，不要直接 reset/clean；
- 大数据、基础模型与音频不进入 Git；实验 JSON、协议和小型 head 应及时拉回本地。

## 20. 文件与证据索引

### 20.1 GitHub 内

- `README.md`：面向公开读者的当前结果说明；
- `results/claim_ledger.json`：每项 claim 的证据状态；
- `results/sota_scope.json`：SOTA 限定范围；
- `results/manifest.json`：结果文件清单与 hash；
- `results/benchmark_identity_audit.json`：400 query / 387 audio 审计；
- `results/e002/`：matched RBEE/SetPO 三种子；
- `results/e004_longneedle/`：NOVA gate、router、公开预测、bootstrap；
- `experiments/protocols/`：不可回写的实验协议；
- `scripts/spantool.py` 与 `scripts/spantool_runtime.py`：SpanTool 核心；
- `tests/`：评测、候选、NOVA、SpanTool 单测。

### 20.2 交接 ZIP 内

- `DEVELOPMENT_HISTORY.md`：本文；
- `chat/CHAT_TRANSCRIPT_SANITIZED.md`：可读聊天记录；
- `chat/CHAT_TRANSCRIPT_SANITIZED.jsonl`：机器可读聊天记录；
- `evidence/server_metadata/`：外部 benchmark、自然背景、SC-PSetPO、SpanTool 的原始 JSON/TXT；
- `weights/setpo_seed1_adapter/`：E002 seed-1 LoRA；
- `weights/spantool_warmstart_head/`：1000-update run 的起始 head；
- `weights/spantool_best_head/`：当前最佳 head；
- `repository_snapshot.bundle`：该分支与 main 的离线 Git bundle；
- `GIT_POINTERS.txt`：远端、分支、commit 与使用说明；
- `ARTIFACT_MANIFEST.sha256`：ZIP 内容校验表。

聊天导出仅包含本地会话数据库中可见的 user/assistant message；系统提示、隐藏 reasoning、工具内部输出不会被导出。SSH 密码、token 等敏感字段已替换为 `[REDACTED]`。聊天记录用于补足决策上下文，不能覆盖原始实验文件。

## 21. 最终状态

截至 2026-09-02：项目已经从“找一个能刷的音频大模型 benchmark”推进到一套可审计的方法与证据链。SpotSound 上的 NOVA 是当前可用点估计 SOTA；RelTwin 上的 RBEE/SetPO 是最可信机制增益；Clotho、AEGBench 与 AudioGrounding 暴露了泛化短板；SpanTool 已把下一版方法从“继续堆 loss/router”升级为真正的结构化输出范式，但公开结果尚未转正。

接手者最重要的不是继续搜索更多模块，而是把 SpanTool 约束成安全的 residual/abstaining boundary tool，并在未触碰、任务定义一致的数据上获得双 benchmark 正结果。做到这一点，项目才能从“有一个谨慎限定的点估计 SOTA”升级为“方法新颖、机制扎实、泛化完整”的 ICASSP 论文。
